"""Dataset interface.

A provider's job is to hand back graphs already in the shape the decomposition expects:
undirected, self-loop free, with the metadata a model and a split strategy need. Everything
dataset-specific -- which PyG class, which preprocessing flag -- stops here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

from cacose.registry import Registry

__all__ = ["GraphBundle", "DatasetProvider", "PROVIDERS", "normalize_graph"]

PROVIDERS: Registry[DatasetProvider] = Registry("dataset providers")


@dataclass
class GraphBundle:
    """What a provider returns: graphs plus everything downstream needs to know."""

    name: str
    task: str  # "nc" or "gc"
    graphs: list[Data]  # exactly one for node classification
    num_features: int
    num_classes: int
    labels: Tensor  # node labels (nc) or one label per graph (gc)
    #: dataset-supplied split masks, when it has them (Geom-GCN ships 10 columns)
    provided_masks: dict[str, Tensor] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_split_units(self) -> int:
        """What a split indexes: nodes for nc, whole graphs for gc."""
        return self.graphs[0].num_nodes if self.task == "nc" else len(self.graphs)

    def __repr__(self) -> str:
        return (
            f"GraphBundle({self.name}, task={self.task}, graphs={len(self.graphs)}, "
            f"features={self.num_features}, classes={self.num_classes}, "
            f"split_units={self.num_split_units})"
        )


def normalize_graph(data: Data) -> Data:
    """Undirected, self-loop free.

    `networkx.core_number` raises on self-loops and the k-core definition assumes a simple
    graph, so this runs before decomposition. GCNConv re-adds self-loops internally, so the
    model is unaffected.
    """
    edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    return data


class DatasetProvider(ABC):
    """Loads one family of datasets."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @abstractmethod
    def load(self, name: str) -> GraphBundle:
        """Fetch (downloading if needed) and normalise."""

    @abstractmethod
    def available(self) -> list[str]:
        """Dataset names this provider understands."""
