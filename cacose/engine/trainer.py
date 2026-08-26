"""Training loop with early stopping.

Deliberately task-agnostic: it calls `Task.train_epoch` and `Task.evaluate` and never asks
whether it is looking at nodes or graphs. Adding a task means adding a `Task`, not editing this.

The stopping rule is the paper's: track the best validation score, stop after `patience` epochs
without improvement, and restore the best weights before touching the test split. Restoring
matters -- reporting the final epoch instead of the best one silently changes the protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from torch import nn

from cacose.eval import Evaluator
from cacose.tasks.base import Task
from cacose.types import Splits

__all__ = ["Trainer", "TrainingHistory"]


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_score: list[float] = field(default_factory=list)
    best_score: float = float("-inf")
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False


class Trainer:
    def __init__(
        self,
        task: Task,
        evaluator: Evaluator,
        *,
        epochs: int,
        patience: int,
        lr: float,
        weight_decay: float,
        optimizer_cls=None,
        verbose: bool = False,
    ) -> None:
        import torch

        self.task, self.evaluator = task, evaluator
        self.epochs, self.patience = epochs, patience
        self.lr, self.weight_decay = lr, weight_decay
        self.optimizer_cls = optimizer_cls or torch.optim.Adam
        self.verbose = verbose

    def fit(self, model: nn.Module, data, splits: Splits) -> TrainingHistory:
        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        history = TrainingHistory()
        best_state = copy.deepcopy(model.state_dict())
        since_improved = 0

        for epoch in range(1, self.epochs + 1):
            loss = self.task.train_epoch(model, data, splits, optimizer)
            val = self.task.predict(model, data, splits.val)
            score = self.evaluator(val.logits, val.targets)[self.evaluator.primary]

            history.train_loss.append(loss)
            history.val_score.append(score)
            history.epochs_run = epoch

            if score > history.best_score:
                history.best_score = score
                history.best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                since_improved = 0
            else:
                since_improved += 1

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  epoch {epoch:4d}  loss {loss:.4f}  val {score:.4f}", flush=True)

            if since_improved >= self.patience:
                history.stopped_early = True
                break

        model.load_state_dict(best_state)  # evaluate the best model, not the last one
        return history
