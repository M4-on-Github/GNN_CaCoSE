"""One (config, seed) -> one RunResult.

This is the only place that knows the whole pipeline: load, decompose, split, build, train,
evaluate, record. `scripts/run.py` is a thin CLI over it, and the SLURM array calls that script
once per seed.
"""

from __future__ import annotations

import random
import subprocess
import time

import numpy as np
import torch

from cacose.config import RunConfig
from cacose.data import PROVIDERS, SPLITS, GraphBundle
from cacose.decompose import DECOMPOSERS, DecompositionCache
from cacose.engine.paths import Paths
from cacose.engine.trainer import Trainer
from cacose.eval import Evaluator
from cacose.nn.model import CaCoSE
from cacose.tasks import TASKS
from cacose.types import GraphSample, RunResult

__all__ = ["ExperimentRunner", "set_seed"]


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from torch_geometric import seed_everything

        seed_everything(seed)
    except ImportError:  # pragma: no cover
        pass
    # warn_only: several PyG kernels have no deterministic implementation, and hard-failing
    # would block the run rather than making it reproducible.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


class ExperimentRunner:
    def __init__(self, cfg: RunConfig, paths: Paths | None = None, device: str | None = None):
        self.cfg = cfg
        self.paths = (paths or Paths.from_env()).ensure()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ------------------------------------------------------------------ stages

    def load_bundle(self) -> GraphBundle:
        provider = PROVIDERS.create(self.cfg.data.provider, root=self.paths.datasets)
        return provider.load(self.cfg.data.dataset)

    def build_samples(self, bundle: GraphBundle) -> tuple[list[GraphSample], list[int]]:
        """Decompose every graph once, caching by (dataset, decomposer, params)."""
        decomposer = DECOMPOSERS.create(
            self.cfg.decompose.decomposer, **self.cfg.decompose.params()
        )
        cache = DecompositionCache(self.paths.cache)

        samples: list[GraphSample] = []
        for i, g in enumerate(bundle.graphs):
            # per-graph cache keys for gc; a single key for the one nc graph
            key = bundle.name if bundle.task == "nc" else f"{bundle.name}#{i}"
            decomp = cache.load_or_compute(key, decomposer, g.edge_index, g.num_nodes)
            y = bundle.labels if bundle.task == "nc" else bundle.labels[i].reshape(1)
            samples.append(GraphSample(x=g.x, decomposition=decomp, y=y).to(self.device))

        ks = sorted({k for s in samples for k in s.decomposition.ks})
        return samples, ks

    def build_model(self, bundle: GraphBundle, ks: list[int]) -> CaCoSE:
        model = CaCoSE.from_config(
            self.cfg, in_dim=bundle.num_features, num_classes=bundle.num_classes, ks=ks
        )
        return model.to(self.device)

    # ------------------------------------------------------------------ entry point

    def run(self, seed: int, *, verbose: bool = False) -> RunResult:
        started = time.time()
        set_seed(seed)

        bundle = self.load_bundle()
        samples, ks = self.build_samples(bundle)
        split = SPLITS.create(self.cfg.data.split, **self.cfg.data.split_args)
        splits = split.make(bundle, seed)

        model = self.build_model(bundle, ks)
        task = (
            TASKS.create("gc", batch_size=self.cfg.train.batch_size)
            if self.cfg.task == "gc"
            else TASKS.create("nc")
        )
        evaluator = Evaluator(["accuracy", "macro_f1"])

        trainer = Trainer(
            task,
            evaluator,
            epochs=self.cfg.train.epochs,
            patience=self.cfg.train.patience,
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            verbose=verbose,
        )
        data = samples[0] if self.cfg.task == "nc" else samples
        history = trainer.fit(model, data, splits)
        scores = task.evaluate(model, data, splits, evaluator)

        total_subgraphs = sum(len(s.decomposition) for s in samples)
        isolated = sum(s.decomposition.num_isolated_nodes() for s in samples)

        import torch_geometric

        return RunResult(
            dataset=bundle.name,
            task=self.cfg.task,
            seed=seed,
            config_hash=self.cfg.config_hash(),
            best_val_acc=float(history.best_score),
            test_acc=float(scores["test"].get("accuracy", float("nan"))),
            epochs_run=history.epochs_run,
            best_epoch=history.best_epoch,
            wall_time_s=round(time.time() - started, 2),
            kmax=max(ks),
            num_subgraphs=total_subgraphs,
            num_isolated_nodes=isolated,
            num_params=model.num_parameters(),
            git_sha=_git_sha(),
            torch_version=torch.__version__,
            pyg_version=torch_geometric.__version__,
            extra={
                "ks": ks,
                "split_sizes": list(splits.sizes),
                "stopped_early": history.stopped_early,
                "device": str(self.device),
                "scores": scores,
                "config": self.cfg.to_dict(),
            },
        )
