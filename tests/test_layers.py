"""Layer tests: readouts, backbones, pooling, cross-subgraph attention.

The registry round-trips matter as much as the numerics here -- a config naming a component
that was never registered should fail loudly at construction, not silently fall back.
"""

from __future__ import annotations

import math

import pytest
import torch

from cacose.nn.attention import CrossSubgraphAttention
from cacose.nn.backbone import BACKBONES, Backbone
from cacose.nn.pooling import POOLINGS, PoolingStrategy
from cacose.nn.readout import READOUTS

torch.manual_seed(0)

# a 4-node path-plus-chord graph, both directions
EDGES = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 3], [1, 2, 3, 3, 0, 1, 2, 0]], dtype=torch.long)
N_NODES, IN_DIM, HIDDEN, D_S = 4, 7, 16, 16


# ---------------------------------------------------------------- readouts


def test_mean_readout_matches_manual_mean():
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    out = READOUTS.create("mean")(x)
    assert torch.allclose(out, torch.tensor([[2.0, 4.0]]))


def test_sum_and_max_readouts():
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    assert torch.allclose(READOUTS.create("sum")(x), torch.tensor([[4.0, 8.0]]))
    assert torch.allclose(READOUTS.create("max")(x), torch.tensor([[3.0, 6.0]]))


def test_meanmax_concatenates_and_doubles_width():
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    r = READOUTS.create("mean+max")
    assert r.width_multiplier == 2
    assert torch.allclose(r(x), torch.tensor([[2.0, 4.0, 3.0, 6.0]]))


@pytest.mark.parametrize("name", READOUTS.names())
def test_readout_respects_batch_vector(name):
    r = READOUTS.create(name)
    x = torch.randn(6, 5)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = r(x, batch)
    assert out.shape == (2, 5 * r.width_multiplier)


# ---------------------------------------------------------------- backbones


@pytest.mark.parametrize("name", BACKBONES.names())
def test_backbone_output_shape(name):
    net = BACKBONES.create(name, in_dim=IN_DIM, hidden=HIDDEN, num_layers=2, dropout=0.0)
    assert isinstance(net, Backbone)
    out = net(torch.randn(N_NODES, IN_DIM), EDGES)
    assert out.shape == (N_NODES, HIDDEN)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_backbone_depth_is_configurable(layers):
    net = BACKBONES.create("gcn", in_dim=IN_DIM, hidden=HIDDEN, num_layers=layers, dropout=0.0)
    assert len(net.convs) == layers
    assert net(torch.randn(N_NODES, IN_DIM), EDGES).shape == (N_NODES, HIDDEN)


def test_backbone_rejects_zero_layers():
    with pytest.raises(ValueError):
        BACKBONES.create("gcn", in_dim=IN_DIM, hidden=HIDDEN, num_layers=0)


def test_sagpool_scores_with_gcn_not_pyg_default_graphconv():
    """The paper scores pooling attention with a GCN; PyG's SAGPooling defaults to GraphConv."""
    from torch_geometric.nn import GCNConv

    pool = POOLINGS.create("sagpool", hidden=HIDDEN, d_s=D_S)
    assert isinstance(pool.pool.gnn, GCNConv)


# ---------------------------------------------------------------- pooling


@pytest.mark.parametrize("name", POOLINGS.names())
def test_pooling_returns_one_d_s_vector(name):
    pool = POOLINGS.create(name, hidden=HIDDEN, d_s=D_S, ratio=0.5)
    assert isinstance(pool, PoolingStrategy)
    out = pool(torch.randn(N_NODES, HIDDEN), EDGES)
    assert out.shape == (1, D_S)


@pytest.mark.parametrize("readout", READOUTS.names())
def test_pooling_projects_every_readout_to_d_s(readout):
    pool = POOLINGS.create("sagpool", hidden=HIDDEN, d_s=D_S, ratio=0.5, readout=readout)
    assert pool(torch.randn(N_NODES, HIDDEN), EDGES).shape == (1, D_S)


def test_projection_is_identity_when_widths_already_match():
    """No pointless Linear when the readout already emits d_s."""
    same = POOLINGS.create("sagpool", hidden=D_S, d_s=D_S, readout="mean")
    assert isinstance(same.project, torch.nn.Identity)
    wider = POOLINGS.create("sagpool", hidden=D_S, d_s=D_S, readout="mean+max")
    assert isinstance(wider.project, torch.nn.Linear)


def test_pooling_handles_a_batch_of_subgraphs():
    pool = POOLINGS.create("sagpool", hidden=HIDDEN, d_s=D_S, ratio=0.5)
    x = torch.randn(2 * N_NODES, HIDDEN)
    edges = torch.cat([EDGES, EDGES + N_NODES], dim=1)
    batch = torch.cat([torch.zeros(N_NODES), torch.ones(N_NODES)]).long()
    assert pool(x, edges, batch).shape == (2, D_S)


# ---------------------------------------------------------------- attention


def test_attention_shapes_and_per_head_weights():
    attn = CrossSubgraphAttention(D_S, num_heads=2)
    z = torch.randn(1, 5, D_S)
    out, w = attn(z)
    assert out.shape == (1, 5, D_S)
    assert w.shape == (1, 2, 5, 5), "per-head weights must survive for the Phase 3 visualiser"


def test_attention_rows_sum_to_one():
    attn = CrossSubgraphAttention(D_S, num_heads=2)
    _, w = attn(torch.randn(3, 6, D_S))
    assert torch.allclose(w.sum(dim=-1), torch.ones(3, 2, 6), atol=1e-5)


def test_attention_equals_softmax_weighted_average_under_identity_projections():
    """With Q=K=V=I and an identity output projection, the layer must reduce to
    softmax(ZZ^T / sqrt(d)) Z -- the paper's Equation (8) written out."""
    d, n = 8, 4
    attn = CrossSubgraphAttention(d, num_heads=1).eval()
    eye = torch.eye(d)
    with torch.no_grad():
        attn.mha.in_proj_weight.copy_(torch.cat([eye, eye, eye], dim=0))
        attn.mha.in_proj_bias.zero_()
        attn.mha.out_proj.weight.copy_(eye)
        attn.mha.out_proj.bias.zero_()

    z = torch.randn(1, n, d)
    out, w = attn(z)

    expected_w = torch.softmax(z[0] @ z[0].T / math.sqrt(d), dim=-1)
    assert torch.allclose(w[0, 0], expected_w, atol=1e-5)
    assert torch.allclose(out[0], expected_w @ z[0], atol=1e-5)


def test_attention_residual_flag_changes_output():
    z = torch.randn(1, 4, D_S)
    plain = CrossSubgraphAttention(D_S, num_heads=1, residual=False).eval()
    withres = CrossSubgraphAttention(D_S, num_heads=1, residual=True).eval()
    withres.load_state_dict(plain.state_dict(), strict=False)
    assert plain.norm is None and withres.norm is not None
    assert not torch.allclose(plain(z)[0], withres(z)[0])


def test_attention_padding_mask_zeroes_padded_keys():
    """Graph classification pads to the widest N_S; padded subgraphs must not be attended to."""
    attn = CrossSubgraphAttention(D_S, num_heads=1).eval()
    z = torch.randn(1, 4, D_S)
    mask = torch.tensor([[False, False, True, True]])  # last two are padding
    _, w = attn(z, key_padding_mask=mask)
    assert torch.allclose(w[0, 0, :, 2:], torch.zeros(4, 2), atol=1e-6)


def test_attention_rejects_bad_shape_and_head_count():
    with pytest.raises(ValueError):
        CrossSubgraphAttention(D_S, num_heads=1)(torch.randn(4, D_S))  # missing batch dim
    with pytest.raises(ValueError):
        CrossSubgraphAttention(15, num_heads=2)  # 15 not divisible by 2


# ---------------------------------------------------------------- registries


@pytest.mark.parametrize(
    ("registry", "kwargs"),
    [
        (READOUTS, {}),
        (BACKBONES, {"in_dim": IN_DIM, "hidden": HIDDEN}),
        (POOLINGS, {"hidden": HIDDEN, "d_s": D_S}),
    ],
    ids=["readouts", "backbones", "poolings"],
)
def test_every_registered_name_constructs(registry, kwargs):
    assert registry.names(), f"{registry.name} is empty"
    for name in registry.names():
        assert isinstance(registry.create(name, **kwargs), torch.nn.Module)


@pytest.mark.parametrize("registry", [READOUTS, BACKBONES, POOLINGS])
def test_unknown_name_raises_with_available_options(registry):
    with pytest.raises(KeyError, match="unknown entry"):
        registry.get("definitely_not_registered")
