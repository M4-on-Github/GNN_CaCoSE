"""Metrics and the evaluator that runs them.

Accuracy is what the paper reports. Macro-F1 is here because the heterophilic datasets are
class-imbalanced enough that accuracy alone can flatter a model, and it costs three lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from cacose.registry import Registry

__all__ = ["Metric", "METRICS", "Accuracy", "MacroF1", "Evaluator"]

METRICS: Registry[Metric] = Registry("metrics")


class Metric(ABC):
    name: str = "metric"

    @abstractmethod
    def __call__(self, logits: Tensor, targets: Tensor) -> float:
        """Higher is better."""


@METRICS.register("accuracy")
class Accuracy(Metric):
    name = "accuracy"

    def __call__(self, logits: Tensor, targets: Tensor) -> float:
        if targets.numel() == 0:
            return float("nan")
        return float((logits.argmax(dim=-1) == targets).float().mean())


@METRICS.register("macro_f1")
class MacroF1(Metric):
    name = "macro_f1"

    def __call__(self, logits: Tensor, targets: Tensor) -> float:
        if targets.numel() == 0:
            return float("nan")
        preds = logits.argmax(dim=-1)
        scores = []
        for cls in torch.unique(targets):
            tp = float(((preds == cls) & (targets == cls)).sum())
            fp = float(((preds == cls) & (targets != cls)).sum())
            fn = float(((preds != cls) & (targets == cls)).sum())
            denom = 2 * tp + fp + fn
            scores.append(0.0 if denom == 0 else 2 * tp / denom)
        return sum(scores) / len(scores) if scores else float("nan")


class Evaluator:
    """Runs a set of metrics and returns them by name."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        names = metrics or ["accuracy"]
        self.metrics = [METRICS.create(n) for n in names]

    @property
    def primary(self) -> str:
        return self.metrics[0].name

    def __call__(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        return {m.name: m(logits, targets) for m in self.metrics}
