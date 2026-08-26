"""Node classification.

One graph, one forward pass per epoch; the split selects which node predictions contribute to
the loss. `data` is a single `GraphSample`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cacose.tasks.base import TASKS, EpochOutput, Task
from cacose.types import GraphSample, Splits

__all__ = ["NodeClassification"]


@TASKS.register("nc")
class NodeClassification(Task):
    name = "nc"

    def train_epoch(
        self, model: nn.Module, data: GraphSample, splits: Splits, optimizer
    ) -> float:
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data)
        idx = splits.train.to(logits.device)
        loss = self.loss_fn(logits[idx], data.y[idx])
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, model: nn.Module, data: GraphSample, indices: Tensor) -> EpochOutput:
        model.eval()
        logits, aux = model(data)
        idx = indices.to(logits.device)
        return EpochOutput(
            loss=self.loss_fn(logits[idx], data.y[idx]).detach(),
            logits=logits[idx],
            targets=data.y[idx],
            aux=aux,
        )
