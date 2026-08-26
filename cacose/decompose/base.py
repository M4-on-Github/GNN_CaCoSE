"""Decomposer interface.

`KCoreCaEF` is the paper's method. The interface exists because the paper's own Figure 5
compares it against Louvain, Metis, hierarchical and random-walk partitions, so a second
implementation is already demanded rather than hypothetical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import torch
from torch import Tensor

from cacose.registry import Registry
from cacose.types import Decomposition, Subgraph

__all__ = ["GraphDecomposer", "DECOMPOSERS", "to_simple_graph", "build_subgraphs"]

DECOMPOSERS: Registry[GraphDecomposer] = Registry("decomposers")


class GraphDecomposer(ABC):
    """Maps a graph to a partition of its edge set into cohesive subgraphs."""

    #: Stable identifier used in the cache key. Set by subclasses.
    id: str = "base"

    @abstractmethod
    def decompose(self, edge_index: Tensor, num_nodes: int) -> Decomposition:
        """Partition the edges of a graph given as a PyG-style `edge_index`."""

    @abstractmethod
    def params(self) -> dict:
        """Parameters that affect the output. Hashed into the cache key."""


def to_simple_graph(edge_index: Tensor, num_nodes: int) -> nx.Graph:
    """PyG `edge_index` -> undirected simple graph.

    Self-loops are dropped: `networkx.core_number` raises on them, and the k-core definition
    assumes a simple graph. GCNConv re-adds self-loops internally, so nothing is lost.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    g.add_edges_from((u, v) for u, v in zip(src, dst, strict=True) if u != v)
    return g


def build_subgraphs(scores: dict[tuple[int, int], int], num_nodes: int) -> Decomposition:
    """Group scored edges into `Subgraph`s, one per distinct score.

    `scores` maps a canonical undirected pair (u < v) to its final score. The emitted
    `edge_index` carries both directions so it is directly usable by a PyG conv.
    """
    by_k: dict[int, list[tuple[int, int]]] = {}
    for (u, v), k in scores.items():
        by_k.setdefault(k, []).append((u, v))

    subgraphs: list[Subgraph] = []
    for k in sorted(by_k):
        pairs = sorted(by_k[k])
        globals_ = sorted({n for pair in pairs for n in pair})
        g2l = {g: i for i, g in enumerate(globals_)}
        # both directions, so the tensor is conv-ready
        src = [g2l[u] for u, _ in pairs] + [g2l[v] for _, v in pairs]
        dst = [g2l[v] for _, v in pairs] + [g2l[u] for u, _ in pairs]
        subgraphs.append(
            Subgraph(
                k=k,
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                node_map=torch.tensor(globals_, dtype=torch.long),
            )
        )

    kmax = max(by_k) if by_k else 0
    return Decomposition(subgraphs=subgraphs, num_nodes=num_nodes, kmax=kmax)
