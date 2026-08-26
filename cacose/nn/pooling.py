"""Subgraph pooling: node embeddings -> one embedding per subgraph.

SAGPool is the paper's choice. Note `GNN=GCNConv`: PyG's `SAGPooling` defaults to `GraphConv`,
but the paper scores nodes with a GCN, so the default would quietly be the wrong model.

TopK / DMoN / GMT are the substitutions the paper's Table 4 ablates; they slot in here as new
registry entries without the model changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling

from cacose.nn.readout import READOUTS, Readout
from cacose.registry import Registry

__all__ = ["PoolingStrategy", "POOLINGS", "SAGPoolStrategy"]

POOLINGS: Registry[PoolingStrategy] = Registry("poolings")


class PoolingStrategy(nn.Module, ABC):
    """Node embeddings of one subgraph -> a single `d_s`-dimensional vector."""

    def __init__(
        self,
        hidden: int,
        d_s: int,
        ratio: float = 0.5,
        readout: str | Readout = "mean+max",
    ) -> None:
        super().__init__()
        self.hidden, self.d_s, self.ratio = hidden, d_s, ratio
        self.readout: Readout = READOUTS.create(readout) if isinstance(readout, str) else readout
        pooled_width = hidden * self.readout.width_multiplier
        # project only when the readout width differs from the target
        self.project: nn.Module = (
            nn.Identity() if pooled_width == d_s else nn.Linear(pooled_width, d_s)
        )

    @abstractmethod
    def select(self, x: Tensor, edge_index: Tensor, batch: Tensor | None) -> tuple[Tensor, Tensor]:
        """Keep a subset of nodes. Returns (kept features, their batch vector)."""

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        kept, kept_batch = self.select(x, edge_index, batch)
        return self.project(self.readout(kept, kept_batch))


@POOLINGS.register("sagpool")
class SAGPoolStrategy(PoolingStrategy):
    def __init__(self, hidden: int, d_s: int, ratio: float = 0.5, readout="mean+max") -> None:
        super().__init__(hidden, d_s, ratio, readout)
        self.pool = SAGPooling(hidden, ratio=ratio, GNN=GCNConv)

    def select(self, x: Tensor, edge_index: Tensor, batch: Tensor | None):
        kept, _, _, kept_batch, _, _ = self.pool(x, edge_index, batch=batch)
        return kept, kept_batch


@POOLINGS.register("topk")
class TopKPoolStrategy(PoolingStrategy):
    """Paper Table 4 ablation."""

    def __init__(self, hidden: int, d_s: int, ratio: float = 0.5, readout="mean+max") -> None:
        super().__init__(hidden, d_s, ratio, readout)
        self.pool = TopKPooling(hidden, ratio=ratio)

    def select(self, x: Tensor, edge_index: Tensor, batch: Tensor | None):
        kept, _, _, kept_batch, _, _ = self.pool(x, edge_index, batch=batch)
        return kept, kept_batch


@POOLINGS.register("none")
class NoPoolStrategy(PoolingStrategy):
    """Readout over every node, no selection. Useful as an ablation floor and for testing."""

    def select(self, x: Tensor, edge_index: Tensor, batch: Tensor | None):
        return x, batch
