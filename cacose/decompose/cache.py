"""On-disk cache for decompositions.

Decomposition is deterministic and pure, so caching it is free correctness-wise and saves
recomputing on every seed of a 10-seed sweep.

The key is `(dataset, decomposer_id, param_hash)`. The design spec originally fixed it as
`(dataset, delta, caef_mode)`, which is specific to the k-core decomposer; this form is a
strict superset that reduces to the same thing for `KCoreCaEF` while leaving room for the
Louvain/Metis comparison the paper's Figure 5 makes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from cacose.decompose.base import GraphDecomposer
from cacose.types import Decomposition, Subgraph

__all__ = ["DecompositionCache", "param_hash"]


def param_hash(params: dict) -> str:
    """Stable short hash of a decomposer's parameters."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


class DecompositionCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def key(self, dataset: str, decomposer: GraphDecomposer) -> str:
        return f"{dataset}__{decomposer.id}__{param_hash(decomposer.params())}"

    def path(self, dataset: str, decomposer: GraphDecomposer) -> Path:
        return self.root / f"{self.key(dataset, decomposer)}.pt"

    def load_or_compute(
        self,
        dataset: str,
        decomposer: GraphDecomposer,
        edge_index: Tensor,
        num_nodes: int,
        *,
        use_cache: bool = True,
    ) -> Decomposition:
        p = self.path(dataset, decomposer)
        if use_cache and p.exists():
            return self._load(p)
        decomp = decomposer.decompose(edge_index, num_nodes)
        if use_cache:
            self._save(p, decomp)
        return decomp

    # -- serialisation ------------------------------------------------------------------
    # Stored as plain tensors/ints rather than pickled dataclasses, so a cache written by an
    # older version still loads if the dataclass gains a field.

    @staticmethod
    def _save(path: Path, decomp: Decomposition) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_nodes": decomp.num_nodes,
            "kmax": decomp.kmax,
            "subgraphs": [
                {"k": sg.k, "edge_index": sg.edge_index, "node_map": sg.node_map}
                for sg in decomp.subgraphs
            ],
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)  # atomic, so a killed job cannot leave a half-written cache

    @staticmethod
    def _load(path: Path) -> Decomposition:
        payload = torch.load(path, weights_only=True)
        return Decomposition(
            subgraphs=[Subgraph(**sg) for sg in payload["subgraphs"]],
            num_nodes=payload["num_nodes"],
            kmax=payload["kmax"],
        )
