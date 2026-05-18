"""Base interface for future sparse subnet optimization methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseSparseMethod:
    """Minimal placeholder interface for sparse subnet methods."""

    config: dict[str, Any] = field(default_factory=dict)

    def setup(self) -> None:
        """Prepare method state.

        TODO: define common state for masks, schedules, logging, and hooks.
        """

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        """Run one method-specific update step.

        TODO: implement in concrete methods.
        """
        raise NotImplementedError("Sparse method step is not implemented yet.")
