"""Decomposition tests.

The Figure 2 fixture below is the same worked example tabulated in the design spec. It is
encoded once in each place on purpose: if the implementation drifts from the spec, this file
fails rather than the drift going unnoticed.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pytest
import torch

from cacose.decompose import DECOMPOSERS, DecompositionCache, KCoreCaEF, to_simple_graph

# --------------------------------------------------------------------------------------
# The paper's Figure 2 graph. v1..v10 are 0-indexed as 0..9.
# K5 on v1..v5, a triangle v4-v6-v7, and a K4 on v7..v10.
# --------------------------------------------------------------------------------------

V1, V2, V3, V4, V5, V6, V7, V8, V9, V10 = range(10)

K5_EDGES = list(itertools.combinations([V1, V2, V3, V4, V5], 2))  # 10
TRIANGLE_EDGES = [(V4, V6), (V6, V7), (V4, V7)]  # 3
K4_EDGES = list(itertools.combinations([V7, V8, V9, V10], 2))  # 6
FIG2_EDGES = K5_EDGES + TRIANGLE_EDGES + K4_EDGES
FIG2_NUM_NODES = 10


def canon(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))


def edges_to_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Both directions, as PyG expects."""
    src = [u for u, _ in edges] + [v for _, v in edges]
    dst = [v for _, v in edges] + [u for u, _ in edges]
    return torch.tensor([src, dst], dtype=torch.long)


@pytest.fixture
def fig2_edge_index() -> torch.Tensor:
    return edges_to_edge_index(FIG2_EDGES)


# --------------------------------------------------------------------------------------
# Figure 2: the spec's worked example, assertion by assertion
# --------------------------------------------------------------------------------------


def test_fig2_core_numbers(fig2_edge_index):
    g = to_simple_graph(fig2_edge_index, FIG2_NUM_NODES)
    core = KCoreCaEF.core_numbers(g)
    for v in (V1, V2, V3, V4, V5):
        assert core[v] == 4
    for v in (V7, V8, V9, V10):
        assert core[v] == 3
    assert core[V6] == 2


def test_fig2_raw_edge_scores(fig2_edge_index):
    g = to_simple_graph(fig2_edge_index, FIG2_NUM_NODES)
    raw = KCoreCaEF.raw_scores(g, KCoreCaEF.core_numbers(g))

    for e in K5_EDGES:
        assert raw[canon(*e)] == 4
    for e in K4_EDGES:
        assert raw[canon(*e)] == 3
    assert raw[canon(V4, V6)] == 2
    assert raw[canon(V6, V7)] == 2
    assert raw[canon(V4, V7)] == 3  # before CaEF


def test_fig2_v4_v7_has_no_support_inside_g3(fig2_edge_index):
    """The decisive case: support must be computed inside G_k, not in the full graph.

    v4 and v7 share the neighbour v6, so full-graph support would be 1 and the edge would
    survive at level 3. But core(v6) = 2, so v6 is absent from G_3 and the support is 0.
    """
    g = to_simple_graph(fig2_edge_index, FIG2_NUM_NODES)
    core = KCoreCaEF.core_numbers(g)
    adj = {n: set(g[n]) for n in g}

    assert adj[V4] & adj[V7] == {V6}  # they do share a neighbour in G
    assert KCoreCaEF.support(adj, core, V4, V7, k=3) == 0  # but not inside G_3
    assert KCoreCaEF.support(adj, core, V4, V7, k=2) == 1  # it reappears at level 2


def test_fig2_caef_demotes_only_v4_v7(fig2_edge_index):
    g = to_simple_graph(fig2_edge_index, FIG2_NUM_NODES)
    core = KCoreCaEF.core_numbers(g)
    raw = KCoreCaEF.raw_scores(g, core)
    adj = {n: set(g[n]) for n in g}
    final = KCoreCaEF(delta=3).apply_caef(adj, core, raw)

    demoted = {e for e in raw if final[e] != raw[e]}
    assert demoted == {canon(V4, V7)}
    assert final[canon(V4, V7)] == 2


def test_fig2_partition(fig2_edge_index):
    decomp = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)

    assert decomp.ks == [2, 3, 4]
    assert decomp.kmax == 4
    sizes = {sg.k: sg.num_undirected_edges for sg in decomp.subgraphs}
    assert sizes == {2: 3, 3: 6, 4: 10}


def test_fig2_s2_contents(fig2_edge_index):
    decomp = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)
    s2 = next(sg for sg in decomp.subgraphs if sg.k == 2)

    local_to_global = s2.node_map.tolist()
    pairs = {
        canon(local_to_global[u], local_to_global[v])
        for u, v in zip(s2.edge_index[0].tolist(), s2.edge_index[1].tolist(), strict=True)
    }
    assert pairs == {canon(V4, V6), canon(V6, V7), canon(V4, V7)}


def test_fig2_v4_membership(fig2_edge_index):
    """v4 is in S_2 and S_4 only -- the paper states z_v4 = h_v4(2) + h_v4(4)."""
    decomp = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)
    assert decomp.node_membership()[V4] == [2, 4]


def test_fig2_no_isolated_nodes(fig2_edge_index):
    decomp = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)
    assert decomp.num_isolated_nodes() == 0


# --------------------------------------------------------------------------------------
# Properties over random graphs
# --------------------------------------------------------------------------------------

RANDOM_SEEDS = list(range(30))


def random_graph_edge_index(seed: int, n: int = 30, p: float = 0.18):
    g = nx.gnp_random_graph(n, p, seed=seed)
    edges = [canon(u, v) for u, v in g.edges()]
    return edges_to_edge_index(edges) if edges else torch.empty(2, 0, dtype=torch.long), g


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_every_edge_lands_in_exactly_one_subgraph(seed):
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")
    decomp = KCoreCaEF(delta=3).decompose(ei, g.number_of_nodes())

    seen: list[tuple[int, int]] = []
    for sg in decomp.subgraphs:
        m = sg.node_map.tolist()
        seen += [
            canon(m[u], m[v])
            for u, v in zip(sg.edge_index[0].tolist(), sg.edge_index[1].tolist(), strict=True)
        ]
    # each undirected edge appears exactly twice (once per direction) and in one subgraph only
    assert len(seen) == 2 * g.number_of_edges()
    assert set(seen) == {canon(u, v) for u, v in g.edges()}


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_node_maps_cover_non_isolated_nodes(seed):
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")
    decomp = KCoreCaEF(delta=3).decompose(ei, g.number_of_nodes())

    covered = {n for sg in decomp.subgraphs for n in sg.node_map.tolist()}
    expected = {n for n in g.nodes() if g.degree(n) > 0}
    assert covered == expected
    assert decomp.num_isolated_nodes() == g.number_of_nodes() - len(expected)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_caef_invariant_single_mode(seed):
    """For raw score k >= delta: zero support at k  <=>  final score is k - 1.

    This is the correct form. The tempting simpler claim -- that every edge remaining at a
    level >= delta has positive support -- is false, because an edge demoted from k lands at
    k-1, which may still be >= delta, and single mode never rechecks it.
    """
    delta = 3
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")

    gg = to_simple_graph(ei, g.number_of_nodes())
    core = KCoreCaEF.core_numbers(gg)
    raw = KCoreCaEF.raw_scores(gg, core)
    adj = {n: set(gg[n]) for n in gg}
    final = KCoreCaEF(delta=delta, caef_mode="single").apply_caef(adj, core, raw)

    for e, raw_k in raw.items():
        if raw_k < delta:
            assert final[e] == raw_k  # untouched below the threshold
            continue
        zero_support = KCoreCaEF.support(adj, core, *e, k=raw_k) == 0
        assert zero_support == (final[e] == raw_k - 1)
        assert final[e] in (raw_k, raw_k - 1)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_large_delta_makes_caef_a_noop(seed):
    """With delta above kmax, nothing is eligible for demotion: pure k-core partition."""
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")

    gg = to_simple_graph(ei, g.number_of_nodes())
    core = KCoreCaEF.core_numbers(gg)
    raw = KCoreCaEF.raw_scores(gg, core)
    big_delta = max(raw.values()) + 1

    decomp = KCoreCaEF(delta=big_delta).decompose(ei, g.number_of_nodes())
    expected_sizes: dict[int, int] = {}
    for k in raw.values():
        expected_sizes[k] = expected_sizes.get(k, 0) + 1
    assert {sg.k: sg.num_undirected_edges for sg in decomp.subgraphs} == expected_sizes


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_cascade_matches_single_when_no_double_demotion(seed):
    delta = 3
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")

    gg = to_simple_graph(ei, g.number_of_nodes())
    core = KCoreCaEF.core_numbers(gg)
    raw = KCoreCaEF.raw_scores(gg, core)
    adj = {n: set(gg[n]) for n in gg}

    single = KCoreCaEF(delta=delta, caef_mode="single").apply_caef(adj, core, raw)
    cascade = KCoreCaEF(delta=delta, caef_mode="cascade").apply_caef(adj, core, raw)

    double_demoted = any(
        raw_k >= delta
        and KCoreCaEF.support(adj, core, *e, k=raw_k) == 0
        and raw_k - 1 >= delta
        and KCoreCaEF.support(adj, core, *e, k=raw_k - 1) == 0
        for e, raw_k in raw.items()
    )
    if double_demoted:
        pytest.skip("this graph has an edge that cascades more than one level")
    assert single == cascade


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_cascade_reaches_a_fixed_point(seed):
    """Cascade's defining property, checkable on every graph without skipping.

    After cascading, an edge either sits below delta (out of scope), has positive support at
    its own level, or has bottomed out at 1. Anything else means the loop stopped early.
    """
    delta = 3
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")

    gg = to_simple_graph(ei, g.number_of_nodes())
    core = KCoreCaEF.core_numbers(gg)
    raw = KCoreCaEF.raw_scores(gg, core)
    adj = {n: set(gg[n]) for n in gg}
    cascade = KCoreCaEF(delta=delta, caef_mode="cascade").apply_caef(adj, core, raw)

    for e, k in cascade.items():
        assert k < delta or k == 1 or KCoreCaEF.support(adj, core, *e, k=k) > 0


def test_cascade_and_single_agree_on_figure_2(fig2_edge_index):
    """On the spec's fixture the two modes coincide: (v4,v7) lands at 2, below delta=3,
    so there is nothing left to cascade into."""
    g = to_simple_graph(fig2_edge_index, FIG2_NUM_NODES)
    core = KCoreCaEF.core_numbers(g)
    raw = KCoreCaEF.raw_scores(g, core)
    adj = {n: set(g[n]) for n in g}

    single = KCoreCaEF(delta=3, caef_mode="single").apply_caef(adj, core, raw)
    cascade = KCoreCaEF(delta=3, caef_mode="cascade").apply_caef(adj, core, raw)
    assert single == cascade


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_cascade_never_scores_above_single(seed):
    """Cascade can only push an edge further down, never up."""
    ei, g = random_graph_edge_index(seed)
    if g.number_of_edges() == 0:
        pytest.skip("empty graph")

    gg = to_simple_graph(ei, g.number_of_nodes())
    core = KCoreCaEF.core_numbers(gg)
    raw = KCoreCaEF.raw_scores(gg, core)
    adj = {n: set(gg[n]) for n in gg}

    single = KCoreCaEF(delta=3, caef_mode="single").apply_caef(adj, core, raw)
    cascade = KCoreCaEF(delta=3, caef_mode="cascade").apply_caef(adj, core, raw)
    assert all(cascade[e] <= single[e] <= raw[e] for e in raw)


# --------------------------------------------------------------------------------------
# Determinism, registry, cache
# --------------------------------------------------------------------------------------


def test_decomposition_is_deterministic(fig2_edge_index):
    a = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)
    b = KCoreCaEF(delta=3).decompose(fig2_edge_index, FIG2_NUM_NODES)
    assert a.ks == b.ks
    for x, y in zip(a.subgraphs, b.subgraphs, strict=True):
        assert torch.equal(x.edge_index, y.edge_index)
        assert torch.equal(x.node_map, y.node_map)


def test_self_loops_are_stripped():
    """networkx.core_number raises on self-loops, so they must not survive conversion."""
    edges = K5_EDGES + [(V1, V1)]
    g = to_simple_graph(edges_to_edge_index(edges), FIG2_NUM_NODES)
    assert nx.number_of_selfloops(g) == 0
    KCoreCaEF.core_numbers(g)  # would raise if a self-loop leaked through


def test_registry_resolves_and_constructs():
    assert "kcore_caef" in DECOMPOSERS
    made = DECOMPOSERS.create("kcore_caef", delta=3, caef_mode="single")
    assert isinstance(made, KCoreCaEF)
    with pytest.raises(KeyError):
        DECOMPOSERS.get("does_not_exist")


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        KCoreCaEF(caef_mode="nonsense")
    with pytest.raises(ValueError):
        KCoreCaEF(delta=0)


def test_cache_round_trip(tmp_path, fig2_edge_index):
    cache = DecompositionCache(tmp_path)
    dec = KCoreCaEF(delta=3)

    first = cache.load_or_compute("fig2", dec, fig2_edge_index, FIG2_NUM_NODES)
    assert cache.path("fig2", dec).exists()
    second = cache.load_or_compute("fig2", dec, fig2_edge_index, FIG2_NUM_NODES)

    assert first.ks == second.ks == [2, 3, 4]
    assert first.kmax == second.kmax
    for a, b in zip(first.subgraphs, second.subgraphs, strict=True):
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.node_map, b.node_map)


def test_cache_key_separates_caef_modes(tmp_path):
    """The spec's original (dataset, delta, caef_mode) key generalises to include the
    decomposer id; distinct params must still land in distinct files."""
    cache = DecompositionCache(tmp_path)
    single = cache.key("cora", KCoreCaEF(delta=3, caef_mode="single"))
    cascade = cache.key("cora", KCoreCaEF(delta=3, caef_mode="cascade"))
    other_delta = cache.key("cora", KCoreCaEF(delta=4, caef_mode="single"))
    assert len({single, cascade, other_delta}) == 3
    assert single.startswith("cora__kcore_caef__")
