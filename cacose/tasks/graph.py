"""Graph classification.

Many small graphs; a split selects whole graphs. `data` is a list of `GraphSample`, and batching
is index chunking rather than PyG collation -- each sample carries its own decomposition, which
a standard collate would flatten and lose. The model handles the ragged subgraph counts with
padding plus a key mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cacose.tasks.base import TASKS, EpochOutput, Task
from cacose.types import GraphSample, Splits

__all__ = ["GraphClassification"]


@TASKS.register("gc")
class GraphClassification(Task):
    name = "gc"

    def __init__(self, loss_fn: nn.Module | None = None, batch_size: int = 32) -> None:
        super().__init__(loss_fn)
        self.batch_size = batch_size

    @staticmethod
    def _targets(samples: Sequence[GraphSample], device) -> Tensor:
        return torch.cat([s.y.reshape(1) for s in samples]).to(device)

    def _batches(self, indices: Tensor, shuffle: bool) -> list[Tensor]:
        order = indices[torch.randperm(indices.numel())] if shuffle else indices
        return list(torch.split(order, self.batch_size))

    def train_epoch(
        self, model: nn.Module, data: Sequence[GraphSample], splits: Splits, optimizer
    ) -> float:
        model.train()
        device = self._device_of(model)
        total, seen = 0.0, 0
        for chunk in self._batches(splits.train, shuffle=True):
            batch = [data[int(i)] for i in chunk]
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = self.loss_fn(logits, self._targets(batch, device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        return total / max(seen, 1)

    @torch.no_grad()
    def predict(
        self, model: nn.Module, data: Sequence[GraphSample], indices: Tensor
    ) -> EpochOutput:
        model.eval()
        device = self._device_of(model)
        logits_all, targets_all, last_aux = [], [], {}
        for chunk in self._batches(indices, shuffle=False):
            batch = [data[int(i)] for i in chunk]
            logits, aux = model(batch)
            logits_all.append(logits)
            targets_all.append(self._targets(batch, device))
            last_aux = aux
        logits_cat = torch.cat(logits_all)
        targets_cat = torch.cat(targets_all)
        return EpochOutput(
            loss=self.loss_fn(logits_cat, targets_cat).detach(),
            logits=logits_cat,
            targets=targets_cat,
            aux=last_aux,
        )
