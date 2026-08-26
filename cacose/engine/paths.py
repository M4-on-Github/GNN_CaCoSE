"""Filesystem layout, resolved in exactly one place.

On the cluster, code lives in /home (quota'd, backed up, read-only at runtime) and everything
written lives in /data. Two environment variables carry that split:

    CACOSE_DATA_ROOT   datasets            default ./data
    CACOSE_OUT         results/logs/cache  default ./

Every other module asks `Paths` rather than joining strings, so pointing a run at a scratch
directory is a one-variable change and nothing leaks into the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Paths"]


@dataclass(frozen=True)
class Paths:
    data_root: Path
    out_root: Path

    def __post_init__(self) -> None:
        # accept plain strings; every caller otherwise has to remember to wrap in Path()
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(self, "out_root", Path(self.out_root).expanduser())

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            data_root=Path(os.environ.get("CACOSE_DATA_ROOT", "./data")).expanduser(),
            out_root=Path(os.environ.get("CACOSE_OUT", ".")).expanduser(),
        )

    @property
    def datasets(self) -> Path:
        return self.data_root

    @property
    def cache(self) -> Path:
        return self.out_root / "cache" / "decompositions"

    @property
    def results(self) -> Path:
        return self.out_root / "results"

    @property
    def logs(self) -> Path:
        return self.out_root / "logs"

    def result_dir(self, dataset: str, config_hash: str) -> Path:
        """One directory per (dataset, configuration).

        Nesting by config hash is what keeps a sweep from overwriting itself: varying
        `caef_mode` produces a different hash, so both variants' ten seeds coexist.
        """
        return self.results / dataset / config_hash

    def result_file(self, dataset: str, config_hash: str, seed: int) -> Path:
        return self.result_dir(dataset, config_hash) / f"seed{seed:02d}.json"

    def ensure(self) -> Paths:
        for p in (self.datasets, self.cache, self.results, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self
