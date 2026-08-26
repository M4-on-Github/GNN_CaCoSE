"""k-core edge scoring with closure-aware edge filtration (CaEF).

The paper's method. Equations and their justification are in the design spec; this module is
the executable form. Two readings that the spec settles and that are easy to get wrong:

* The edge score `max{k : (u,v) in G_k}` equals `min(core(u), core(v))`, because an edge
  survives to level k exactly when both endpoints do. No `G_k` needs to be materialised.
* Triadic support is counted **inside `G_k`**, not in the full graph. Since `G_k` is the
  node-induced subgraph on `{v : core(v) >= k}`, that is just the full-graph common
  neighbourhood filtered by core number -- see `support`.
"""

from __future__ import annotations

import math

import networkx as nx
from torch import Tensor

from cacose.decompose.base import DECOMPOSERS, GraphDecomposer, build_subgraphs, to_simple_graph
from cacose.types import Decomposition

__all__ = ["KCoreCaEF"]

Pair = tuple[int, int]


@DECOMPOSERS.register("kcore_caef")
class KCoreCaEF(GraphDecomposer):
    id = "kcore_caef"

    def __init__(self, delta: int = 3, caef_mode: str = "single") -> None:
        if caef_mode not in ("single", "cascade"):
            raise ValueError(f"caef_mode must be 'single' or 'cascade', got {caef_mode!r}")
        if delta < 1:
            raise ValueError(f"delta must be >= 1, got {delta}")
        self.delta = delta
        self.caef_mode = caef_mode

    def params(self) -> dict:
        return {"delta": self.delta, "caef_mode": self.caef_mode}

    # -- stages, exposed individually so tests can assert at each one ------------------

    @staticmethod
    def core_numbers(g: nx.Graph) -> dict[int, int]:
        return nx.core_number(g)

    @staticmethod
    def raw_scores(g: nx.Graph, core: dict[int, int]) -> dict[Pair, int]:
        """C(u,v) = min(core(u), core(v)), keyed by the canonical pair u < v."""
        return {(min(u, v), max(u, v)): min(core[u], core[v]) for u, v in g.edges()}

    @staticmethod
    def support(adj: dict[int, set[int]], core: dict[int, int], u: int, v: int, k: int) -> int:
        """|N_k(u) ∩ N_k(v)| -- common neighbours that survive to level k."""
        shared = adj[u] & adj[v]
        return sum(1 for w in shared if core[w] >= k)

    def apply_caef(
        self, adj: dict[int, set[int]], core: dict[int, int], raw: dict[Pair, int]
    ) -> dict[Pair, int]:
        """Demote unsupported edges.

        `single` takes at most one step down (the paper's Algorithm 1 shows one assignment);
        `cascade` keeps stepping while the edge is still at or above delta and still has no
        support. Both are the same loop with a different step budget.
        """
        budget = 1 if self.caef_mode == "single" else math.inf
        out: dict[Pair, int] = {}
        for (u, v), raw_k in raw.items():
            k, steps = raw_k, 0
            while k >= self.delta and steps < budget and self.support(adj, core, u, v, k) == 0:
                k = max(k - 1, 1)
                steps += 1
                if k == 1:  # nothing below level 1 to demote into
                    break
            out[(u, v)] = k
        return out

    # -- entry point ------------------------------------------------------------------

    def decompose(self, edge_index: Tensor, num_nodes: int) -> Decomposition:
        g = to_simple_graph(edge_index, num_nodes)
        core = self.core_numbers(g)
        raw = self.raw_scores(g, core)
        adj = {n: set(g[n]) for n in g}
        final = self.apply_caef(adj, core, raw)
        return build_subgraphs(final, num_nodes)

    def __repr__(self) -> str:
        return f"KCoreCaEF(delta={self.delta}, caef_mode={self.caef_mode!r})"
