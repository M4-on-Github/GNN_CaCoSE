"""Graph readout: node features -> one vector per graph.

Which readout the paper uses is spec ambiguity #4 -- it says only "READOUT". `mean+max` is the
default because that is standard SAGPool-g practice; the alternatives are registered so the
Milestone 6 sweep can vary this without touching the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from cacose.registry import Registry

__all__ = ["Readout", "READOUTS"]

READOUTS: Registry[Readout] = Registry("readouts")


class Readout(nn.Module, ABC):
    #: output width as a multiple of the input width
    width_multiplier: int = 1

    @abstractmethod
    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """[N, d] node features -> [B, d * width_multiplier]."""


def _batch_or_single(x: Tensor, batch: Tensor | None) -> Tensor:
    return torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch


@READOUTS.register("mean")
class MeanReadout(Readout):
    width_multiplier = 1

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        return global_mean_pool(x, _batch_or_single(x, batch))


@READOUTS.register("max")
class MaxReadout(Readout):
    width_multiplier = 1

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        return global_max_pool(x, _batch_or_single(x, batch))


@READOUTS.register("sum")
class SumReadout(Readout):
    width_multiplier = 1

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        return global_add_pool(x, _batch_or_single(x, batch))


@READOUTS.register("mean+max")
class MeanMaxReadout(Readout):
    width_multiplier = 2

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        b = _batch_or_single(x, batch)
        return torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=-1)
