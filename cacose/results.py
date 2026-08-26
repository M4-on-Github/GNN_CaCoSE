"""Writing, finding and aggregating run results.

Results are the one artefact that survives a job, so the layout matters:
`results/<dataset>/<config_hash>/seed<NN>.json`. Grouping by config hash means a sweep that
varies one flag cannot overwrite the run it is being compared against.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from cacose.engine.paths import Paths
from cacose.types import RunResult

__all__ = ["ResultStore", "Aggregate"]


@dataclass
class Aggregate:
    dataset: str
    config_hash: str
    n_seeds: int
    mean: float
    std: float
    seeds: list[int]

    def format_row(self, target: float | None = None) -> str:
        # ASCII: SLURM logs and Windows consoles mangle non-ASCII under cp1252
        cell = f"{self.mean * 100:.2f} +/- {self.std * 100:.2f}"
        if target is None:
            return f"| {self.dataset} | {self.config_hash} | {self.n_seeds} | {cell} | - | - |"
        delta = self.mean * 100 - target
        return (
            f"| {self.dataset} | {self.config_hash} | {self.n_seeds} | {cell} | "
            f"{target:.2f} | {delta:+.2f} |"
        )


class ResultStore:
    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or Paths.from_env()

    def write(self, result: RunResult) -> Path:
        path = self.paths.result_file(result.dataset, result.config_hash, result.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)  # atomic: a killed job cannot leave a truncated result
        return path

    def load_all(self, dataset: str | None = None) -> list[dict]:
        root = self.paths.results
        if not root.exists():
            return []
        pattern = f"{dataset}/*/seed*.json" if dataset else "*/*/seed*.json"
        out = []
        for p in sorted(root.glob(pattern)):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                print(f"  skipping unreadable result: {p}")
        return out

    def aggregate(self, dataset: str | None = None, metric: str = "test_acc") -> list[Aggregate]:
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in self.load_all(dataset):
            groups.setdefault((row["dataset"], row["config_hash"]), []).append(row)

        aggregates = []
        for (ds, cfg_hash), rows in sorted(groups.items()):
            values = [float(r[metric]) for r in rows]
            aggregates.append(
                Aggregate(
                    dataset=ds,
                    config_hash=cfg_hash,
                    n_seeds=len(values),
                    mean=statistics.fmean(values),
                    # population std is undefined for one sample; report 0 rather than crashing
                    std=statistics.stdev(values) if len(values) > 1 else 0.0,
                    seeds=sorted(int(r["seed"]) for r in rows),
                )
            )
        return aggregates
