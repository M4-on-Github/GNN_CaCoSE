"""Config, splits, metrics, paths and results.

These run without touching the network -- dataset downloads belong in the smoke run, not the
unit suite, so CI stays offline-safe.
"""

from __future__ import annotations

import json

import pytest
import torch

from cacose.config import RunConfig
from cacose.data.base import GraphBundle
from cacose.data.splits import SPLITS
from cacose.engine.paths import Paths
from cacose.eval import Evaluator
from cacose.results import ResultStore
from cacose.types import RunResult


def bundle(n=200, classes=4, task="nc"):
    labels = torch.arange(n) % classes
    graphs = [type("G", (), {"num_nodes": n})()]
    return GraphBundle(
        name="synthetic",
        task=task,
        graphs=graphs,
        num_features=8,
        num_classes=classes,
        labels=labels,
    )


# ---------------------------------------------------------------- config


def test_config_defaults_match_the_paper():
    cfg = RunConfig()
    assert (cfg.train.lr, cfg.train.weight_decay) == (2.5e-3, 1.0e-4)
    assert (cfg.model.hidden, cfg.model.d_s, cfg.model.pool_ratio) == (128, 128, 0.5)
    assert cfg.decompose.delta == 3


def test_config_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("task: nc\nmodel:\n  hidden: 128\n  dropout: 0.5\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel:\n  dropout: 0.1\n")
    cfg = RunConfig.from_yaml(tmp_path / "child.yaml")
    assert cfg.model.dropout == 0.1  # overridden
    assert cfg.model.hidden == 128  # inherited


def test_unknown_config_key_is_rejected(tmp_path):
    """A typo in a sweep config must fail loudly, not be silently ignored."""
    (tmp_path / "bad.yaml").write_text("model:\n  hiden: 64\n")
    with pytest.raises(ValueError, match="unknown key"):
        RunConfig.from_yaml(tmp_path / "bad.yaml")


def test_config_hash_ignores_seeds_but_tracks_everything_else():
    a, b = RunConfig(), RunConfig()
    b.seeds = [99]
    assert a.config_hash() == b.config_hash(), "seeds must not change the hash"
    b.model.readout = "sum"
    assert a.config_hash() != b.config_hash(), "a swept flag must change the hash"


@pytest.mark.parametrize("name", ["cora", "chameleon", "mutag"])
def test_shipped_configs_load(name):
    cfg = RunConfig.from_yaml(f"configs/{name}.yaml")
    assert cfg.data.dataset == name
    assert cfg.model.num_heads == (1 if cfg.task == "gc" else 2)


# ---------------------------------------------------------------- splits


def test_fixed_count_split_uses_exact_counts_and_leaves_a_remainder():
    """Cora's 1208/500/500 covers 2208 of 2708 nodes; the surplus is intentional."""
    b = bundle(n=2708)
    s = SPLITS.create("fixed_count", train=1208, val=500, test=500).make(b, seed=0)
    assert s.sizes == (1208, 500, 500)
    assert sum(s.sizes) == 2208


def test_fixed_count_rejects_impossible_request():
    with pytest.raises(ValueError, match="split needs"):
        SPLITS.create("fixed_count", train=100, val=100, test=100).make(bundle(n=50), seed=0)


def test_splits_are_disjoint_and_deterministic():
    b = bundle()
    make = lambda seed: SPLITS.create(  # noqa: E731
        "stratified_ratio", train=0.48, val=0.32, test=0.20
    ).make(b, seed)
    a, again, other = make(0), make(0), make(1)

    assert torch.equal(a.train, again.train), "same seed must give the same split"
    assert not torch.equal(a.train, other.train), "different seeds must differ"
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(getattr(a, x).tolist()) & set(getattr(a, y).tolist()))


def test_stratified_split_preserves_class_balance():
    b = bundle(n=400, classes=4)
    s = SPLITS.create("stratified_ratio", train=0.5, val=0.25, test=0.25).make(b, seed=0)
    for part in (s.train, s.val, s.test):
        counts = torch.bincount(b.labels[part], minlength=4)
        assert counts.min() > 0, "every class must appear in every split"
        assert int(counts.max() - counts.min()) <= 1


def test_stratified_split_is_not_class_ordered():
    """Building class by class leaves splits sorted by label; that must be shuffled away.

    Otherwise any unshuffled consumer sees single-class batches, and slicing a split for a
    diagnostic quietly samples one class -- which is exactly how a real debugging session here
    got misled.
    """
    b = bundle(n=300, classes=3)
    s = SPLITS.create("stratified_ratio", train=0.6, val=0.2, test=0.2).make(b, seed=0)
    head = b.labels[s.train][:40]
    assert head.unique().numel() > 1, "the head of a split must not be one class"


def test_provided_masks_require_the_dataset_to_ship_them():
    with pytest.raises(ValueError, match="ships no split masks"):
        SPLITS.create("provided_masks").make(bundle(), seed=0)


def test_provided_masks_select_the_seed_column():
    b = bundle(n=10)
    cols = 3
    masks = {}
    for i, split in enumerate(("train", "val", "test")):
        m = torch.zeros(10, cols, dtype=torch.bool)
        m[i * 3 : i * 3 + 3, :] = True
        masks[split] = m
    b.provided_masks = masks
    s = SPLITS.create("provided_masks").make(b, seed=1)
    assert s.sizes == (3, 3, 3)


# ---------------------------------------------------------------- metrics


def test_accuracy_and_macro_f1():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    scores = Evaluator(["accuracy", "macro_f1"])(logits, targets)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert 0.0 < scores["macro_f1"] <= 1.0


def test_macro_f1_punishes_a_collapsed_predictor_more_than_accuracy():
    """The exact failure seen on MUTAG: predicting one class scores well on accuracy alone."""
    n = 100
    targets = torch.cat([torch.zeros(70), torch.ones(30)]).long()
    logits = torch.zeros(n, 2)
    logits[:, 0] = 1.0  # always predict the majority class
    scores = Evaluator(["accuracy", "macro_f1"])(logits, targets)
    assert scores["accuracy"] == pytest.approx(0.70)
    assert scores["macro_f1"] < 0.45


def test_metrics_on_empty_input_are_nan_not_a_crash():
    scores = Evaluator(["accuracy", "macro_f1"])(torch.zeros(0, 2), torch.zeros(0).long())
    assert all(v != v for v in scores.values())


# ---------------------------------------------------------------- paths and results


def test_paths_accept_strings_and_nest_results_by_config_hash(tmp_path):
    p = Paths(data_root=str(tmp_path / "d"), out_root=str(tmp_path / "o"))
    assert p.result_file("cora", "abc123", 7).name == "seed07.json"
    assert p.result_file("cora", "abc123", 7).parent.name == "abc123"
    p.ensure()
    assert p.cache.exists() and p.results.exists() and p.logs.exists()


def test_result_store_round_trip_and_aggregate(tmp_path):
    store = ResultStore(Paths(tmp_path / "d", tmp_path / "o"))
    for seed, acc in enumerate([0.80, 0.84, 0.82]):
        store.write(
            RunResult(
                dataset="cora", task="nc", seed=seed, config_hash="deadbeef", test_acc=acc
            )
        )
    aggs = store.aggregate("cora")
    assert len(aggs) == 1
    assert aggs[0].n_seeds == 3
    assert aggs[0].mean == pytest.approx(0.82)
    assert aggs[0].seeds == [0, 1, 2]


def test_two_configs_do_not_overwrite_each_other(tmp_path):
    """The whole point of nesting by hash: a sweep must not clobber its own baseline."""
    store = ResultStore(Paths(tmp_path / "d", tmp_path / "o"))
    store.write(RunResult(dataset="cora", task="nc", seed=0, config_hash="aaa", test_acc=0.1))
    store.write(RunResult(dataset="cora", task="nc", seed=0, config_hash="bbb", test_acc=0.9))
    assert len(store.aggregate("cora")) == 2


def test_result_json_matches_the_documented_schema(tmp_path):
    store = ResultStore(Paths(tmp_path / "d", tmp_path / "o"))
    path = store.write(RunResult(dataset="cora", task="nc", seed=0, config_hash="x"))
    loaded = json.loads(path.read_text())
    for key in ("dataset", "seed", "config_hash", "test_acc", "kmax", "num_subgraphs"):
        assert key in loaded
