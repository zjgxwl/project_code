"""Loss-landscape and interpolation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn


def _state_interpolate(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if set(first) != set(second):
        raise ValueError("State dictionaries must have identical keys.")
    return {
        name: (1.0 - alpha) * first[name].detach() + alpha * second[name].detach()
        for name in first
    }


@torch.no_grad()
def _average_loss(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> float:
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = int(targets.size(0))
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("dataloader yielded no samples.")
    return total_loss / total_samples


def linear_interpolation_losses(
    model: nn.Module,
    first_state: Mapping[str, torch.Tensor],
    second_state: Mapping[str, torch.Tensor],
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device | str,
    alphas: Sequence[float],
    max_batches: int | None = None,
) -> list[float]:
    """Evaluate validation loss along a linear interpolation path."""
    resolved_device = torch.device(device)
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    was_training = model.training
    losses: list[float] = []
    try:
        model.eval()
        for alpha in alphas:
            interpolated = _state_interpolate(first_state, second_state, float(alpha))
            model.load_state_dict(interpolated, strict=True)
            losses.append(_average_loss(model, dataloader, criterion, resolved_device, max_batches))
    finally:
        model.load_state_dict(original_state, strict=True)
        model.train(was_training)
    return losses


def loss_barrier(losses: Sequence[float]) -> float:
    """Return the interpolation loss barrier from a sequence of path losses."""
    if len(losses) < 2:
        raise ValueError("loss_barrier requires at least two losses.")
    endpoints = (float(losses[0]) + float(losses[-1])) / 2.0
    return max(float(loss) for loss in losses) - endpoints
