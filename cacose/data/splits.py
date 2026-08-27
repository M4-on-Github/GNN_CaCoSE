"""Split strategies.

The paper specifies three different conventions, so this is an interface rather than a function:
fixed counts for Cora, stratified ratios elsewhere, and the dataset's own masks as the
alternative reading of ambiguity #7.

Every strategy is seeded and deterministic: the same (seed, dataset) always yields the same
split, which is what makes a 10-seed mean reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from cacose.data.base import GraphBundle
from cacose.registry import Registry
from cacose.types import Splits

__all__ = ["SplitStrategy", "SPLITS", "FixedCountSplit", "StratifiedRatioSplit", "KFoldSplit"]

SPLITS: Registry[SplitStrategy] = Registry("split strategies")


class SplitStrategy(ABC):
    @abstractmethod
    def make(self, bundle: GraphBundle, seed: int) -> Splits:
        """Indices into nodes (nc) or graphs (gc)."""

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        return torch.Generator().manual_seed(int(seed))


@SPLITS.register("fixed_count")
class FixedCountSplit(SplitStrategy):
    """Exact counts, as the paper gives for Cora (1208 train / 500 val / 500 test).

    Note those three sum to 2208 of Cora's 2708 nodes, leaving 500 unused. That is what the
    paper's text says (spec ambiguity #6b), so it is what this implements; the surplus is
    reported rather than quietly folded into training.
    """

    def __init__(self, train: int, val: int, test: int) -> None:
        self.train, self.val, self.test = int(train), int(val), int(test)

    def make(self, bundle: GraphBundle, seed: int) -> Splits:
        n = bundle.num_split_units
        needed = self.train + self.val + self.test
        if needed > n:
            raise ValueError(f"split needs {needed} units but {bundle.name} has {n}")
        perm = torch.randperm(n, generator=self._generator(seed))
        val = perm[: self.val]
        test = perm[self.val : self.val + self.test]
        train = perm[self.val + self.test : self.val + self.test + self.train]
        return Splits(train=train, val=val, test=test)


@SPLITS.register("stratified_ratio")
class StratifiedRatioSplit(SplitStrategy):
    """Proportional and class-balanced -- the paper's 48/32/20 and 80/10/10 conventions.

    Stratifying matters on the small heterophilic datasets and on MUTAG, where an unstratified
    draw can leave a class almost absent from a split and make seeds incomparable.
    """

    def __init__(self, train: float, val: float, test: float) -> None:
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1, got {total}")
        self.train, self.val, self.test = train, val, test

    def make(self, bundle: GraphBundle, seed: int) -> Splits:
        labels = bundle.labels
        gen = self._generator(seed)
        train_idx, val_idx, test_idx = [], [], []

        for cls in labels.unique().tolist():
            members = (labels == cls).nonzero(as_tuple=True)[0]
            members = members[torch.randperm(members.numel(), generator=gen)]
            n = members.numel()
            n_train = int(round(self.train * n))
            n_val = int(round(self.val * n))
            # give the remainder to test so the three always partition the class exactly
            train_idx.append(members[:n_train])
            val_idx.append(members[n_train : n_train + n_val])
            test_idx.append(members[n_train + n_val :])

        # Shuffle after concatenating. Building class by class leaves each split sorted by
        # label, which is a trap: any consumer that iterates a split without shuffling gets
        # single-class batches, and slicing a split for a diagnostic silently samples one class.
        def shuffled(parts: list[Tensor]) -> Tensor:
            idx = torch.cat(parts)
            return idx[torch.randperm(idx.numel(), generator=gen)]

        return Splits(train=shuffled(train_idx), val=shuffled(val_idx), test=shuffled(test_idx))


@SPLITS.register("provided_masks")
class ProvidedMaskSplit(SplitStrategy):
    """The dataset's own splits, one column per seed.

    For Chameleon these are Geom-GCN's 60/20/20 columns, which differ from the paper's stated
    48/32/20 -- the reason this is a selectable strategy rather than the default.
    """

    def make(self, bundle: GraphBundle, seed: int) -> Splits:
        if not bundle.provided_masks:
            raise ValueError(f"{bundle.name} ships no split masks; use another split strategy")

        def column(mask: Tensor) -> Tensor:
            col = mask if mask.dim() == 1 else mask[:, seed % mask.size(1)]
            return col.nonzero(as_tuple=True)[0]

        return Splits(
            train=column(bundle.provided_masks["train"]),
            val=column(bundle.provided_masks["val"]),
            test=column(bundle.provided_masks["test"]),
        )


@SPLITS.register("kfold")
class KFoldSplit(SplitStrategy):
    """Stratified k-fold cross-validation: the seed selects which fold is held out.

    Why this exists. The paper states 80/10/10 for graph classification, but its reported MUTAG
    accuracy of 76.99 cannot be produced that way. With 188 graphs a 10% test split is 18-19
    graphs, so every per-seed accuracy is k/n for integer k and any 10-seed mean must be a
    multiple of 1/(10n) -- 76.99 is not such a multiple for any n in that range. Under k-fold the
    folds have different sizes (19 x 9 + 17), so the mean of per-fold accuracies can reach values
    a fixed split cannot. k-fold is also the standard TUDataset protocol used by most of the
    baselines the paper compares against.

    The fold partition is fixed by `shuffle_seed` and is identical across runs; the run seed picks
    the fold. Ten runs therefore cover every graph exactly once as test, which is what makes the
    mean far more stable than ten independent 10% draws.
    """

    def __init__(self, folds: int = 10, shuffle_seed: int = 0) -> None:
        if folds < 3:
            raise ValueError(f"need at least 3 folds to carve out train/val/test, got {folds}")
        self.folds, self.shuffle_seed = int(folds), int(shuffle_seed)

    def _fold_assignment(self, labels: Tensor) -> Tensor:
        """Stratified fold id per unit, dealt round-robin within each class."""
        gen = self._generator(self.shuffle_seed)
        assignment = torch.empty(labels.numel(), dtype=torch.long)
        for cls in labels.unique().tolist():
            members = (labels == cls).nonzero(as_tuple=True)[0]
            members = members[torch.randperm(members.numel(), generator=gen)]
            assignment[members] = torch.arange(members.numel()) % self.folds
        return assignment

    def make(self, bundle: GraphBundle, seed: int) -> Splits:
        assignment = self._fold_assignment(bundle.labels)
        test_fold = seed % self.folds
        val_fold = (test_fold + 1) % self.folds  # the next fold validates, the rest train
        return Splits(
            train=((assignment != test_fold) & (assignment != val_fold)).nonzero(as_tuple=True)[0],
            val=(assignment == val_fold).nonzero(as_tuple=True)[0],
            test=(assignment == test_fold).nonzero(as_tuple=True)[0],
        )
