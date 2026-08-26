"""End-to-end model tests for both tasks, plus the `aux` contract Phase 3 depends on."""

from __future__ import annotations

import itertools

import pytest
import torch

from cacose.decompose import KCoreCaEF
from cacose.nn.model import CaCoSE
from cacose.tasks import TASKS
from cacose.types import GraphSample, Splits
from tests.test_decompose import FIG2_EDGES, FIG2_NUM_NODES, edges_to_edge_index

IN_DIM, HIDDEN, D_S, N_CLASSES = 6, 8, 8, 3
AUX_KEYS = {"Z_S", "Z_S_attn", "attn_weights", "subgraph_ids"}


def fig2_sample(num_classes=N_CLASSES, seed=0):
    torch.manual_seed(seed)
    decomp = KCoreCaEF(delta=3).decompose(edges_to_edge_index(FIG2_EDGES), FIG2_NUM_NODES)
    x = torch.randn(FIG2_NUM_NODES, IN_DIM)
    y = torch.randint(0, num_classes, (FIG2_NUM_NODES,))
    return GraphSample(x=x, decomposition=decomp, y=y), decomp


def _clique(nodes):
    return list(itertools.combinations(nodes, 2))


# Shapes chosen so the subgraph counts genuinely differ (1, 2, 3). A regular graph -- a ring with
# chords, say -- yields one score class for every size, which would make a "ragged batch" test
# silently vacuous.
SHAPES = {
    "triangle": (3, _clique([0, 1, 2])),  # -> 1 subgraph,  ks [2]
    "k4_pendant": (6, _clique([0, 1, 2, 3]) + [(3, 4), (4, 5)]),  # -> 2, ks [1, 3]
    "k4_k3_chain": (
        9,
        _clique([0, 1, 2, 3]) + [(3, 4), (4, 5)] + _clique([5, 6, 7]) + [(7, 8)],
    ),  # -> 3, ks [1, 2, 3]
    "k5_k4_bridge": (
        10,
        _clique([0, 1, 2, 3, 4]) + [(3, 5), (5, 6), (3, 6)] + _clique([6, 7, 8, 9]),
    ),  # -> 3, ks [2, 3, 4]
}


def small_graph_sample(shape, seed, num_classes=2):
    n_nodes, edges = SHAPES[shape]
    g = torch.Generator().manual_seed(seed)
    decomp = KCoreCaEF(delta=3).decompose(edges_to_edge_index(edges), n_nodes)
    return GraphSample(
        x=torch.randn(n_nodes, IN_DIM, generator=g),
        decomposition=decomp,
        y=torch.randint(0, num_classes, (1,), generator=g),
    )


def build(task, ks, **kw):
    return CaCoSE(
        in_dim=IN_DIM,
        num_classes=N_CLASSES if task == "nc" else 2,
        ks=ks,
        task=task,
        hidden=HIDDEN,
        d_s=D_S,
        num_heads=kw.pop("num_heads", 2 if task == "nc" else 1),
        dropout=kw.pop("dropout", 0.0),
        **kw,
    )


# ------------------------------------------------------------------ node classification


def test_nc_forward_shapes():
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks)
    logits, aux = model(sample)
    assert logits.shape == (FIG2_NUM_NODES, N_CLASSES)
    assert torch.isfinite(logits).all()
    assert aux["Z_S"].shape == (1, len(decomp), D_S)


def test_nc_backward_reaches_every_branch():
    """Every k present must receive gradient -- a silently dead branch would still train."""
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks)
    logits, _ = model(sample)
    torch.nn.functional.cross_entropy(logits, sample.y).backward()

    for k in decomp.ks:
        for name, mod in (("backbone", model.backbones), ("pooler", model.poolers)):
            grads = [p.grad for p in mod[str(k)].parameters() if p.requires_grad]
            assert grads, f"{name} k={k} has no parameters"
            assert any(g is not None and g.abs().sum() > 0 for g in grads), (
                f"{name} for k={k} received no gradient"
            )


def test_nc_loss_is_finite_and_decreases_on_a_few_steps():
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks)
    task = TASKS.create("nc")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    splits = Splits(
        train=torch.arange(FIG2_NUM_NODES),
        val=torch.arange(FIG2_NUM_NODES),
        test=torch.arange(FIG2_NUM_NODES),
    )
    first = task.train_epoch(model, sample, splits, opt)
    for _ in range(15):
        last = task.train_epoch(model, sample, splits, opt)
    assert all(map(torch.isfinite, map(torch.tensor, (first, last))))
    assert last < first, "the model should be able to overfit 10 nodes"


def test_nc_rejects_multiple_graphs():
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks)
    with pytest.raises(ValueError, match="one graph"):
        model([sample, sample])


# ------------------------------------------------------------------ graph classification


def test_gc_forward_over_a_ragged_batch():
    """Graphs with different subgraph counts must batch without the padding leaking in."""
    samples = [
        small_graph_sample(sh, seed=i)
        for i, sh in enumerate(["triangle", "k4_pendant", "k5_k4_bridge"])
    ]
    ks = sorted({k for s in samples for k in s.decomposition.ks})
    assert {len(s.decomposition) for s in samples} == {1, 2, 3}, "batch must be ragged"

    model = build("gc", ks)
    logits, aux = model(samples)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert aux["Z_G"].shape == (3, D_S)


def test_gc_padding_does_not_change_a_graphs_prediction():
    """A graph batched with wider neighbours must predict the same as when batched alone."""
    torch.manual_seed(0)
    narrow = small_graph_sample("triangle", seed=1)
    wide = small_graph_sample("k5_k4_bridge", seed=2)
    ks = sorted(set(narrow.decomposition.ks) | set(wide.decomposition.ks))

    model = build("gc", ks).eval()
    with torch.no_grad():
        alone, _ = model([narrow])
        together, _ = model([narrow, wide])
    assert torch.allclose(alone[0], together[0], atol=1e-5)


def test_gc_backward_and_epoch():
    samples = [
        small_graph_sample(sh, seed=i)
        for i, sh in enumerate(["triangle", "k4_pendant", "k4_k3_chain", "k5_k4_bridge"])
    ]
    ks = sorted({k for s in samples for k in s.decomposition.ks})
    model = build("gc", ks)
    task = TASKS.create("gc", batch_size=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    splits = Splits(train=torch.arange(4), val=torch.arange(4), test=torch.arange(4))

    loss = task.train_epoch(model, samples, splits, opt)
    assert torch.isfinite(torch.tensor(loss))


def test_gc_module_dict_must_span_the_dataset_not_one_graph():
    """Building from a single graph's ks is the obvious bug; it must fail loudly."""
    samples = [
        small_graph_sample(sh, seed=i) for i, sh in enumerate(["triangle", "k5_k4_bridge"])
    ]
    ks_all = sorted({k for s in samples for k in s.decomposition.ks})
    ks_one = samples[0].decomposition.ks
    if set(ks_one) == set(ks_all):
        pytest.skip("this pair happens to share all k values")

    model = build("gc", ks_one)
    with pytest.raises(KeyError, match="union of k"):
        model(samples)


# ------------------------------------------------------------------ aux contract (Phase 3)


def test_aux_carries_exactly_the_contracted_keys():
    sample, decomp = fig2_sample()
    _, aux = build("nc", decomp.ks)(sample)
    assert set(aux) >= AUX_KEYS, f"missing {AUX_KEYS - set(aux)}"


def test_aux_attention_weights_are_per_head():
    """Averaged heads would satisfy a shape check but destroy what the visualiser shows."""
    sample, decomp = fig2_sample()
    heads = 2
    _, aux = build("nc", decomp.ks, num_heads=heads)(sample)
    n_sub = len(decomp)
    assert aux["attn_weights"].shape == (1, heads, n_sub, n_sub)


def test_aux_subgraph_ids_identify_each_row_of_z_s():
    sample, decomp = fig2_sample()
    _, aux = build("nc", decomp.ks)(sample)
    assert aux["subgraph_ids"][0].tolist() == decomp.ks


def test_aux_subgraph_ids_mark_padding_with_minus_one():
    samples = [
        small_graph_sample(sh, seed=i) for i, sh in enumerate(["triangle", "k5_k4_bridge"])
    ]
    ks = sorted({k for s in samples for k in s.decomposition.ks})
    _, aux = build("gc", ks)(samples)

    ids = aux["subgraph_ids"]
    for gi, s in enumerate(samples):
        real = len(s.decomposition)
        assert ids[gi, :real].tolist() == s.decomposition.ks
        assert (ids[gi, real:] == -1).all()


# ------------------------------------------------------------------ configuration matrix


@pytest.mark.parametrize(
    ("share_weights", "attn_residual"), list(itertools.product([False, True], repeat=2))
)
def test_ablation_flags_all_run(share_weights, attn_residual):
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks, share_weights=share_weights, attn_residual=attn_residual)
    logits, aux = model(sample)
    assert torch.isfinite(logits).all()
    expected_branches = 1 if share_weights else len(decomp.ks)
    assert len(model.backbones) == expected_branches


@pytest.mark.parametrize("backbone", ["gcn", "gat", "sage"])
@pytest.mark.parametrize("pooling", ["sagpool", "topk"])
def test_backbone_and_pooling_substitutions_run(backbone, pooling):
    """Paper Table 4 ablations must be a config change, not a code change."""
    sample, decomp = fig2_sample()
    model = build("nc", decomp.ks, backbone=backbone, pooling=pooling)
    logits, _ = model(sample)
    assert torch.isfinite(logits).all()


def test_share_weights_reduces_parameter_count():
    _, decomp = fig2_sample()
    separate = build("nc", decomp.ks, share_weights=False).num_parameters()
    shared = build("nc", decomp.ks, share_weights=True).num_parameters()
    assert shared < separate


def test_empty_decomposition_is_rejected():
    with pytest.raises(ValueError, match="no subgraphs"):
        build("nc", [])
