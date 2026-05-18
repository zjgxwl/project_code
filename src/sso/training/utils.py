"""Training utility helpers."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible debug runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve a configured device name to a torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
