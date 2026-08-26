"""Feature-combination tests.

The headline assertion is the paper's own worked example: v4 belongs to S_2 and S_4, so its final
representation must be the exact sum of its two per-subgraph rows.
"""

from __future__ import annotations

import pytest
import torch

from cacose.decompose import KCoreCaEF
from cacose.nn.merge import merge_node_features
from tests.test_decompose import FIG2_EDGES, FIG2_NUM_NODES, V4, V6, edges_to_edge_index

HIDDEN, D_S = 4, 3


@pytest.fixture
def fig2_decomposition():
    return KCoreCaEF(delta=3).decompose(edges_to_edge_index(FIG2_EDGES), FIG2_NUM_NODES)


def synthetic_blocks(decomp, hidden=HIDDEN, d_s=D_S):
    """Distinct constant blocks per subgraph, so contributions are traceable by eye."""
    h = [torch.full((sg.num_nodes, hidden), float(i + 1)) for i, sg in enumerate(decomp.subgraphs)]
    z = torch.stack([torch.full((d_s,), float(10 * (i + 1))) for i in range(len(decomp))])
    return h, z


def test_shape_is_hidden_plus_d_s(fig2_decomposition):
    h, z = synthetic_blocks(fig2_decomposition)
    out = merge_node_features(h, z, fig2_decomposition)
    assert out.shape == (FIG2_NUM_NODES, HIDDEN + D_S)


def test_v4_is_the_exact_sum_of_its_two_subgraph_rows(fig2_decomposition):
    """The paper states z_v4 = h_v4(2) + h_v4(4). v4 is in S_2 and S_4, not S_3."""
    decomp = fig2_decomposition
    h, z = synthetic_blocks(decomp)
    out = merge_node_features(h, z, decomp)

    contributions = []
    for i, sg in enumerate(decomp.subgraphs):
        nodes = sg.node_map.tolist()
        if V4 in nodes:
            local = nodes.index(V4)
            contributions.append(torch.cat([h[i][local], z[i]]))

    assert len(contributions) == 2, "v4 should appear in exactly two subgraphs"
    assert torch.allclose(out[V4], contributions[0] + contributions[1])


def test_node_in_one_subgraph_equals_that_single_row(fig2_decomposition):
    """v6 has core number 2 and sits only in S_2."""
    decomp = fig2_decomposition
    h, z = synthetic_blocks(decomp)
    out = merge_node_features(h, z, decomp)

    hits = [
        (i, sg.node_map.tolist().index(V6))
        for i, sg in enumerate(decomp.subgraphs)
        if V6 in sg.node_map.tolist()
    ]
    assert len(hits) == 1
    i, local = hits[0]
    assert torch.allclose(out[V6], torch.cat([h[i][local], z[i]]))


def test_subgraph_embedding_is_broadcast_to_every_member_node(fig2_decomposition):
    """Each node's tail block is the sum of the Z vectors of the subgraphs it belongs to."""
    decomp = fig2_decomposition
    h, z = synthetic_blocks(decomp)
    out = merge_node_features(h, z, decomp)

    membership = decomp.node_membership()
    k_to_row = {sg.k: z[i] for i, sg in enumerate(decomp.subgraphs)}
    for node, ks in membership.items():
        expected_tail = torch.stack([k_to_row[k] for k in ks]).sum(0)
        assert torch.allclose(out[node, HIDDEN:], expected_tail)


def test_isolated_nodes_stay_zero():
    """A node touched by no edge is in no subgraph and must fall through as zeros."""
    edges = [(0, 1), (1, 2), (0, 2)]
    ei = edges_to_edge_index(edges)
    decomp = KCoreCaEF(delta=3).decompose(ei, num_nodes=5)  # nodes 3 and 4 are isolated

    h, z = synthetic_blocks(decomp)
    out = merge_node_features(h, z, decomp)
    assert torch.equal(out[3], torch.zeros(HIDDEN + D_S))
    assert torch.equal(out[4], torch.zeros(HIDDEN + D_S))
    assert decomp.num_isolated_nodes() == 2


def test_mismatched_inputs_are_rejected(fig2_decomposition):
    h, z = synthetic_blocks(fig2_decomposition)
    with pytest.raises(ValueError, match="feature blocks"):
        merge_node_features(h[:-1], z, fig2_decomposition)
    with pytest.raises(ValueError, match="z_attn"):
        merge_node_features(h, z[:-1], fig2_decomposition)


def test_merge_is_differentiable(fig2_decomposition):
    """index_add_ must not detach the graph -- gradients have to reach the backbones."""
    decomp = fig2_decomposition
    h = [
        torch.randn(sg.num_nodes, HIDDEN, requires_grad=True) for sg in decomp.subgraphs
    ]
    z = torch.randn(len(decomp), D_S, requires_grad=True)

    merge_node_features(h, z, decomp).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    for block in h:
        assert block.grad is not None and torch.isfinite(block.grad).all()
