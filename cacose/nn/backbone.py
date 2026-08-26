"""Per-subgraph message-passing backbone.

The paper uses a GCN. GAT and GraphSAGE are registered because the paper's own Table 4 ablates
exactly that substitution; having them here means the ablation is a config change rather than
a code change.

Depth is spec ambiguity #9 -- the paper does not state it, and two layers is the default in the
GCN work it cites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from cacose.registry import Registry

__all__ = ["Backbone", "BACKBONES", "GCNBackbone"]

BACKBONES: Registry[Backbone] = Registry("backbones")


class Backbone(nn.Module, ABC):
    """Node features + edges -> node embeddings, within one subgraph."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_dim, self.hidden, self.num_layers, self.dropout = in_dim, hidden, num_layers, dropout
        self.convs = nn.ModuleList(self._build_convs())

    @abstractmethod
    def _build_convs(self) -> list[nn.Module]:
        """One conv per layer, first mapping in_dim -> hidden and the rest hidden -> hidden."""

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # no activation or dropout after the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _dims(self) -> list[tuple[int, int]]:
        return [
            (self.in_dim if i == 0 else self.hidden, self.hidden) for i in range(self.num_layers)
        ]


@BACKBONES.register("gcn")
class GCNBackbone(Backbone):
    def _build_convs(self) -> list[nn.Module]:
        return [GCNConv(i, o) for i, o in self._dims()]


@BACKBONES.register("gat")
class GATBackbone(Backbone):
    """Single-head GAT, so the output width matches GCN and the rest of the model is unchanged."""

    def _build_convs(self) -> list[nn.Module]:
        return [GATConv(i, o, heads=1) for i, o in self._dims()]


@BACKBONES.register("sage")
class SAGEBackbone(Backbone):
    def _build_convs(self) -> list[nn.Module]:
        return [SAGEConv(i, o) for i, o in self._dims()]
