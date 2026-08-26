"""Task interface.

Node and graph classification differ in more places than a single flag can cover: what a split
indexes (nodes vs whole graphs), whether a forward pass sees one graph or a batch of them, and
which readout feeds the head. Putting that behind `Task` is what lets `Trainer` run an epoch
without ever asking which task it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from cacose.registry import Registry
from cacose.types import Splits

__all__ = ["Task", "TASKS", "EpochOutput"]

TASKS: Registry[Task] = Registry("tasks")


class EpochOutput:
    """Loss plus whatever the caller may want to inspect, including `aux` for Phase 3."""

    __slots__ = ("loss", "logits", "targets", "aux")

    def __init__(self, loss: Tensor, logits: Tensor, targets: Tensor, aux: dict) -> None:
        self.loss, self.logits, self.targets, self.aux = loss, logits, targets, aux


class Task(ABC):
    name: str = "base"

    def __init__(self, loss_fn: nn.Module | None = None) -> None:
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    @abstractmethod
    def train_epoch(self, model: nn.Module, data, splits: Splits, optimizer) -> float:
        """One optimisation pass over the training split. Returns mean loss."""

    @abstractmethod
    def predict(self, model: nn.Module, data, indices: Tensor) -> EpochOutput:
        """Forward pass over `indices` with no gradient. Used for val and test."""

    def evaluate(self, model: nn.Module, data, splits: Splits, evaluator) -> dict[str, dict]:
        """Metrics for each split. Task-agnostic once `predict` exists."""
        out: dict[str, dict] = {}
        for split_name in ("train", "val", "test"):
            idx = getattr(splits, split_name)
            if idx.numel() == 0:
                out[split_name] = {}
                continue
            res = self.predict(model, data, idx)
            out[split_name] = evaluator(res.logits, res.targets)
        return out

    @staticmethod
    def _device_of(model: nn.Module) -> torch.device:
        return next(model.parameters()).device
