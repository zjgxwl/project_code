"""Classification metrics shared by experiment runners."""

from __future__ import annotations

import torch


def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> dict[str, float]:
    """Return top-k accuracies for a batch of logits."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {tuple(logits.shape)}.")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {tuple(targets.shape)}.")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size.")
    if not topk:
        return {}

    max_k = max(int(k) for k in topk)
    if max_k <= 0:
        raise ValueError("top-k values must be positive.")
    max_k = min(max_k, int(logits.shape[1]))

    _, predictions = logits.topk(max_k, dim=1)
    correct = predictions.eq(targets.view(-1, 1))
    batch_size = max(1, int(targets.numel()))

    accuracies: dict[str, float] = {}
    for k in topk:
        bounded_k = min(int(k), max_k)
        value = correct[:, :bounded_k].any(dim=1).float().sum().item() / batch_size
        accuracies[f"top{int(k)}"] = float(value)
    return accuracies
