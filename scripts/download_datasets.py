"""Download datasets ahead of time, on a machine that has network access.

    python -m scripts.download_datasets --config configs/cora.yaml
    python -m scripts.download_datasets --all

Compute nodes usually have no outbound network, so PyG's download-on-first-use would fail
inside the array job -- after the job has already been queued and scheduled. Running this once
on the login node turns that class of failure into an immediate, obvious one.

This only downloads and decompresses; it does not train, so it is light enough for head1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cacose.config import RunConfig
from cacose.data import PROVIDERS
from cacose.engine import Paths

DEFAULT_CONFIGS = ["configs/cora.yaml", "configs/chameleon.yaml", "configs/mutag.yaml"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", action="append", help="config to prefetch; repeatable")
    p.add_argument("--all", action="store_true", help="prefetch every config in configs/")
    p.add_argument("--data-root", help="override $CACOSE_DATA_ROOT")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configs = DEFAULT_CONFIGS if args.all else (args.config or DEFAULT_CONFIGS)

    paths = Paths.from_env()
    if args.data_root:
        paths = Paths(data_root=Path(args.data_root), out_root=paths.out_root)
    paths.ensure()
    print(f"dataset root: {paths.datasets.resolve()}\n")

    failures = []
    for cfg_path in configs:
        cfg = RunConfig.from_yaml(cfg_path)
        label = f"{cfg.data.dataset} ({cfg.data.provider})"
        try:
            provider = PROVIDERS.create(cfg.data.provider, root=paths.datasets)
            bundle = provider.load(cfg.data.dataset)
            print(
                f"  OK   {label:28s} graphs={len(bundle.graphs):5d} "
                f"features={bundle.num_features:5d} classes={bundle.num_classes}"
            )
        except Exception as exc:  # noqa: BLE001 - report every dataset, fail at the end
            print(f"  FAIL {label:28s} {type(exc).__name__}: {exc}")
            failures.append(label)

    if failures:
        print(f"\n{len(failures)} dataset(s) failed: {', '.join(failures)}")
        print("Datasets must be present before submitting; compute nodes cannot download.")
        return 1

    total = sum(f.stat().st_size for f in paths.datasets.rglob("*") if f.is_file())
    print(f"\nall datasets present ({total / 1e6:.1f} MB under {paths.datasets.resolve()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
