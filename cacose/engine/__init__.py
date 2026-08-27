"""Engine: paths, training loop, experiment runner.

`Paths` is imported eagerly because it is pure stdlib. `Trainer`, `TrainingHistory`,
`ExperimentRunner` and `set_seed` are resolved on first access instead, because they pull in
torch: aggregating results must work on a machine that has none -- a login node with only the
JSON files -- and importing this package eagerly would make that impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cacose.engine.paths import Paths

if TYPE_CHECKING:  # for type checkers and IDEs, never at runtime
    from cacose.engine.runner import ExperimentRunner, set_seed
    from cacose.engine.trainer import Trainer, TrainingHistory

__all__ = ["Paths", "Trainer", "TrainingHistory", "ExperimentRunner", "set_seed"]

_LAZY = {
    "Trainer": "cacose.engine.trainer",
    "TrainingHistory": "cacose.engine.trainer",
    "ExperimentRunner": "cacose.engine.runner",
    "set_seed": "cacose.engine.runner",
}


def __getattr__(name: str):
    """PEP 562 module-level lazy attribute access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
