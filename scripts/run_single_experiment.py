"""Run one (config, seed).

    python -m scripts.run_single_experiment --config configs/cora.yaml --seed 0

The SLURM array calls this once per seed, passing $SLURM_ARRAY_TASK_ID.
"""

from __future__ import annotations

import argparse
import sys

from cacose.config import RunConfig
from cacose.engine import ExperimentRunner, Paths
from cacose.results import ResultStore


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, help="override the config's epoch budget (for smoke runs)")
    p.add_argument("--device", help="cpu or cuda; default picks cuda when available")
    p.add_argument("--out", help="override $CACOSE_OUT for this run")
    p.add_argument("--data-root", help="override $CACOSE_DATA_ROOT for this run")
    p.add_argument("--no-write", action="store_true", help="run but do not persist the result")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
        cfg.train.patience = max(args.epochs, 1)

    paths = Paths.from_env()
    if args.out or args.data_root:
        from pathlib import Path

        paths = Paths(
            data_root=Path(args.data_root) if args.data_root else paths.data_root,
            out_root=Path(args.out) if args.out else paths.out_root,
        )

    print(f"config : {cfg.describe()}")
    print(f"seed   : {args.seed}")
    print(f"out    : {paths.out_root.resolve()}")

    runner = ExperimentRunner(cfg, paths=paths, device=args.device)
    result = runner.run(args.seed, verbose=args.verbose)

    print(
        f"done   : test_acc={result.test_acc:.4f} best_val={result.best_val_acc:.4f} "
        f"epochs={result.epochs_run} (best {result.best_epoch}) "
        f"kmax={result.kmax} subgraphs={result.num_subgraphs} "
        f"params={result.num_params:,} {result.wall_time_s}s"
    )
    if result.num_isolated_nodes:
        print(f"note   : {result.num_isolated_nodes} node(s) in no subgraph -> zero embedding")

    if not args.no_write:
        print(f"wrote  : {ResultStore(paths).write(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
