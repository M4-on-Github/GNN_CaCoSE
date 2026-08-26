"""Feature combination: subgraph embeddings back onto nodes.

The paper's node readout concatenates each node's local embedding with its subgraph's attended
embedding, then *sums* over every subgraph the node belongs to. Summation is what makes a node
that sits in several cohesive regions carry information from all of them -- and it is why the
decomposition partitions edges rather than nodes.

`index_add_` does the accumulation in one shot per subgraph. Nodes in no subgraph keep a zero
row; they are counted rather than hidden, since a large count means the decomposition dropped
part of the graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from cacose.types import Decomposition

__all__ = ["merge_node_features"]


def merge_node_features(
    per_subgraph_h: Sequence[Tensor],
    z_attn: Tensor,
    decomposition: Decomposition,
) -> Tensor:
    """h_v(k) = [H_k[local_v] || Z_k_attn], then z_v = sum over the subgraphs containing v.

    `per_subgraph_h[i]` is [n_i, hidden] for `decomposition.subgraphs[i]`; `z_attn` is
    [num_subgraphs, d_s]. Returns [num_nodes, hidden + d_s].
    """
    n_sub = len(decomposition.subgraphs)
    if len(per_subgraph_h) != n_sub:
        raise ValueError(f"got {len(per_subgraph_h)} feature blocks for {n_sub} subgraphs")
    if z_attn.size(0) != n_sub:
        raise ValueError(f"z_attn has {z_attn.size(0)} rows for {n_sub} subgraphs")

    hidden = per_subgraph_h[0].size(-1) if n_sub else 0
    width = hidden + z_attn.size(-1)
    out = torch.zeros(
        decomposition.num_nodes, width, dtype=z_attn.dtype, device=z_attn.device
    )

    for i, sg in enumerate(decomposition.subgraphs):
        h = per_subgraph_h[i]
        # broadcast the subgraph's single attended vector across its nodes
        z = z_attn[i].unsqueeze(0).expand(h.size(0), -1)
        out.index_add_(0, sg.node_map.to(out.device), torch.cat([h, z], dim=-1))
    return out
