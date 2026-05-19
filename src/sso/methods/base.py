"""Base interface for sparse subnet optimization methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseSparseMethod:
    """Common lifecycle hooks shared by sparse subnet methods.

    Concrete methods can override any hook they need. The defaults are no-ops
    so training loops can call the lifecycle consistently without checking for
    every optional method.
    """

    config: dict[str, Any] = field(default_factory=dict)
    is_setup: bool = field(default=False, init=False)

    def setup(self, **_kwargs: Any) -> "BaseSparseMethod":
        """Prepare method state and return ``self`` for fluent construction."""
        self.is_setup = True
        return self

    def teardown(self) -> None:
        """Release method-owned resources after a run."""
        self.is_setup = False

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        """Run one method-specific update step.

        Most current methods are driven by training-loop hooks, so the default
        step is intentionally a no-op.
        """

    def before_train(self) -> None:
        """Hook called before an epoch or bounded training pass."""

    def after_backward(self) -> None:
        """Hook called after ``loss.backward()`` and before optimizer step."""

    def after_optimizer_step(self) -> None:
        """Hook called after ``optimizer.step()``."""

    def state_dict(self) -> dict[str, Any]:
        """Return method metadata that is independent from model weights."""
        return {
            "config": dict(self.config),
            "is_setup": self.is_setup,
        }
