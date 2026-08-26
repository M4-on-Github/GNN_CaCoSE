"""A tiny name -> class registry.

Every swappable family in the package (decomposers, dataset providers, split strategies,
pooling, backbones, readouts, metrics) uses one of these, so a config string like
`pooling: sagpool` resolves to a class without an if/elif chain anywhere.

The point is Phase 2 and the paper's own ablations: adding Louvain decomposition or swapping
SAGPool for GMT should be a new file plus a decorator, not an edit to the model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

__all__ = ["Registry"]


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator: `@REGISTRY.register("kcore_caef")`."""

        def wrap(cls: type[T]) -> type[T]:
            k = key.lower()
            if k in self._entries:
                raise KeyError(f"{self.name}: '{k}' is already registered to {self._entries[k]!r}")
            self._entries[k] = cls
            return cls

        return wrap

    def get(self, key: str) -> type[T]:
        k = key.lower()
        if k not in self._entries:
            raise KeyError(
                f"{self.name}: unknown entry '{key}'. Available: {', '.join(self.names()) or '(none)'}"
            )
        return self._entries[k]

    def create(self, key: str, **kwargs) -> T:
        return self.get(key)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={self.names()})"
