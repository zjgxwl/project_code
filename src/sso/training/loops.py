"""Minimal training and evaluation loops."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Iterable

import torch
from torch import nn


def _to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    method: Any | None = None,
    epoch: int = 0,
    total_epochs: int = 1,
    batch_callback: Callable[[Mapping[str, Any]], None] | None = None,
    log_interval: int = 0,
) -> dict[str, float]:
    """Run one bounded training epoch."""
    model.train()
    if method is not None:
        method.before_train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    try:
        total_batches = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        total_batches = None

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_started_at = time.perf_counter()
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        task_loss = criterion(logits, targets)
        loss = task_loss
        if method is not None and hasattr(method, "regularization_loss"):
            loss = loss + method.regularization_loss(epoch=epoch, total_epochs=total_epochs)
        loss.backward()
        if method is not None:
            if hasattr(method, "after_backward"):
                method.after_backward()
            elif hasattr(method, "mask_gradients"):
                method.mask_gradients()
        optimizer.step()
        if method is not None and hasattr(method, "after_optimizer_step"):
            method.after_optimizer_step()

        batch_size = targets.size(0)
        batch_loss = _to_float(task_loss)
        batch_correct = int((logits.argmax(dim=1) == targets).sum().item())
        batch_acc = 0.0 if batch_size == 0 else batch_correct / batch_size
        total_loss += _to_float(task_loss) * batch_size
        total_correct += batch_correct
        total_samples += batch_size
        if batch_callback is not None and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            running_loss = 0.0 if total_samples == 0 else total_loss / total_samples
            running_acc = 0.0 if total_samples == 0 else total_correct / total_samples
            batch_callback(
                {
                    "epoch": epoch + 1,
                    "epochs": total_epochs,
                    "batch": batch_idx + 1,
                    "batches": total_batches,
                    "batch_loss": batch_loss,
                    "batch_acc": batch_acc,
                    "running_train_loss": running_loss,
                    "running_train_acc": running_acc,
                    "batch_size": int(batch_size),
                    "duration_sec": time.perf_counter() - batch_started_at,
                }
            )

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    method: Any | None = None,
) -> dict[str, float]:
    """Run a bounded evaluation pass."""
    model.eval()
    if method is not None and hasattr(method, "apply_mask"):
        method.apply_mask()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += _to_float(loss) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}
