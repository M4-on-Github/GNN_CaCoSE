"""Typed configuration.

YAML files are thin: `configs/base.yaml` holds every default, and a dataset file overrides only
what differs. The dataclasses below are the real schema -- an unknown key raises rather than being
silently ignored, which is the failure mode that wastes a whole sweep.

`config_hash` deliberately excludes the seed, so all ten seeds of one configuration land in the
same results directory and the aggregator can group them without parsing filenames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["RunConfig", "DataConfig", "DecomposeConfig", "ModelConfig", "TrainConfig"]


def _build(cls, data: dict, where: str):
    """Construct a dataclass, rejecting unknown keys with a message that names them."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in '{where}': {sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    return cls(**data)


@dataclass
class DataConfig:
    dataset: str = "cora"
    provider: str = "planetoid"
    split: str = "fixed_count"
    #: arguments for the split strategy, e.g. counts or ratios
    split_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecomposeConfig:
    decomposer: str = "kcore_caef"
    delta: int = 3
    caef_mode: str = "single"

    def params(self) -> dict:
        return {"delta": self.delta, "caef_mode": self.caef_mode}


@dataclass
class ModelConfig:
    hidden: int = 128
    d_s: int = 128
    gcn_layers: int = 2
    backbone: str = "gcn"
    pooling: str = "sagpool"
    readout: str = "mean+max"
    share_weights: bool = False
    pool_ratio: float = 0.5
    num_heads: int = 2
    attn_residual: bool = False
    dropout: float = 0.5


@dataclass
class TrainConfig:
    lr: float = 2.5e-3
    weight_decay: float = 1.0e-4
    epochs: int = 250
    patience: int = 50
    batch_size: int = 32


@dataclass
class RunConfig:
    task: str = "nc"
    data: DataConfig = field(default_factory=DataConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: list[int] = field(default_factory=lambda: list(range(10)))

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_dict(cls, raw: dict) -> RunConfig:
        raw = dict(raw)
        raw.pop("extends", None)
        sections = {
            "data": DataConfig,
            "decompose": DecomposeConfig,
            "model": ModelConfig,
            "train": TrainConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            kwargs[name] = _build(klass, raw.pop(name, {}) or {}, name)
        for key in ("task", "seeds"):
            if key in raw:
                kwargs[key] = raw.pop(key)
        if raw:
            raise ValueError(f"unknown top-level key(s): {sorted(raw)}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a config, following a single `extends:` chain relative to the file itself."""
        merged = _load_with_extends(Path(path))
        return cls.from_dict(merged)

    # ---------------------------------------------------------------- identity

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self, length: int = 8) -> str:
        """Stable hash of everything except the seed list."""
        payload = self.to_dict()
        payload.pop("seeds", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:length]

    def describe(self) -> str:
        return (
            f"{self.data.dataset}/{self.task} "
            f"[{self.config_hash()}] "
            f"delta={self.decompose.delta} caef={self.decompose.caef_mode} "
            f"backbone={self.model.backbone} pool={self.model.pooling} "
            f"readout={self.model.readout} share={self.model.share_weights} "
            f"heads={self.model.num_heads} residual={self.model.attn_residual}"
        )


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular 'extends' involving {path}")
    _seen.add(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.get("extends")
    if not parent:
        return raw
    return _deep_merge(_load_with_extends(path.parent / parent, _seen), raw)
