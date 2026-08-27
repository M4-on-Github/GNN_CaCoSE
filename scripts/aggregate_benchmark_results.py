"""Aggregate results into a markdown table, diffed against the paper.

    python -m scripts.aggregate_benchmark_results
    python -m scripts.aggregate_benchmark_results --dataset chameleon
"""

from __future__ import annotations

import argparse
import sys

from cacose.engine.paths import Paths
from cacose.results import ResultStore

# Paper Tables 1 and 2, and the tolerance Phase 1 accepts (see the design spec, section 1).
TARGETS = {
    "cora": (85.00, 83.5),
    "chameleon": (68.99, 66.5),
    "mutag": (76.99, 74.0),
    "citeseer": (69.42, None),
    "squirrel": (58.86, None),
    "texas": (54.47, None),
    "proteins": (71.79, None),
    "imdb-b": (73.20, None),
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="restrict to one dataset")
    p.add_argument("--metric", default="test_acc")
    p.add_argument("--out", help="override $CACOSE_OUT")
    # Deliberately imports nothing that needs torch, so results can be aggregated on a login
    # node or any machine holding the JSON files -- no container required.
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = Paths.from_env()
    if args.out:
        from pathlib import Path

        paths = Paths(data_root=paths.data_root, out_root=Path(args.out))

    store = ResultStore(paths)
    aggregates = store.aggregate(args.dataset, metric=args.metric)
    if not aggregates:
        print(f"no results under {paths.results.resolve()}")
        return 1

    print(f"\n### CaCoSE reproduction - {args.metric}\n")
    print("| dataset | config | seeds | mean +/- std | paper | delta |")
    print("|---|---|---:|---|---:|---:|")
    for agg in aggregates:
        target = TARGETS.get(agg.dataset, (None, None))[0]
        print(agg.format_row(target))

    print()
    for agg in aggregates:
        target, accept = TARGETS.get(agg.dataset, (None, None))
        missing = [s for s in range(10) if s not in agg.seeds]
        if missing:
            print(f"note: {agg.dataset} [{agg.config_hash}] missing seeds {missing}")
        if accept is not None and agg.n_seeds >= 10:
            verdict = "PASS" if agg.mean * 100 >= accept else "BELOW TARGET"
            print(
                f"gate: {agg.dataset} [{agg.config_hash}] {agg.mean * 100:.2f} "
                f"vs accept >= {accept} -> {verdict}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
