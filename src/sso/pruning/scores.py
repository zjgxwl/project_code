"""Pruning score functions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .params import iter_prunable_named_parameters


def magnitude_score(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return absolute weight magnitude scores for prunable parameters."""
    return {
        name: parameter.detach().abs().clone()
        for name, parameter in iter_prunable_named_parameters(model)
    }


def snip_score(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
    """Placeholder for future SNIP scores."""
    raise NotImplementedError("TODO: implement SNIP scoring in a future chapter method.")


def grasp_score(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
    """Placeholder for future GraSP scores."""
    raise NotImplementedError("TODO: implement GraSP scoring in a future chapter method.")


def synflow_score(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
    """Placeholder for future SynFlow scores."""
    raise NotImplementedError("TODO: implement SynFlow scoring in a future chapter method.")
