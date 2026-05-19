"""Reusable multi-epoch training runner for thesis experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .loops import evaluate, train_one_epoch


EpochMetricsFn = Callable[[int, int], Mapping[str, Any]]


@dataclass(frozen=True)
class FitResult:
    """Summary returned by ``fit_model``."""

    epochs: int
    final_train: dict[str, float]
    final_val: dict[str, float]
    best_val_accuracy: float
    best_epoch: int


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the first optimizer group learning rate."""
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _method_metrics(method: Any | None, epoch: int, total_epochs: int) -> dict[str, Any]:
    if method is None or not hasattr(method, "method_metrics"):
        return {}
    metrics = method.method_metrics(epoch=epoch, total_epochs=total_epochs)
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def fit_model(
    *,
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    method: Any | None = None,
    scheduler: Any | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    eval_interval: int = 1,
    checkpoint_interval: int = 0,
    logger: Any | None = None,
    extra_epoch_metrics: EpochMetricsFn | None = None,
    base_metrics: Mapping[str, Any] | None = None,
    log_batches: bool = False,
    log_interval: int = 0,
    log_to_console: bool = True,
) -> FitResult:
    """Train and evaluate a model for a configurable number of epochs."""
    total_epochs = max(1, int(epochs))
    eval_every = max(1, int(eval_interval))
    checkpoint_every = max(0, int(checkpoint_interval))
    final_train: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
    final_val: dict[str, float] = {"loss": 0.0, "accuracy": 0.0}
    best_val_accuracy = float("-inf")
    best_epoch = 0
    common_metrics = dict(base_metrics or {})

    for epoch in range(total_epochs):
        started_at = time.perf_counter()
        lr_before = current_lr(optimizer)

        def batch_callback(record: Mapping[str, Any]) -> None:
            batch_record = {**common_metrics, **dict(record), "lr": lr_before}
            if logger is not None:
                logger.log_batch(batch_record)
            if log_to_console:
                batches = batch_record.get("batches") or "?"
                print(
                    "Epoch "
                    f"[{batch_record['epoch']}/{total_epochs}] "
                    f"Batch [{batch_record['batch']}/{batches}] "
                    f"train_loss={batch_record['running_train_loss']:.4f} "
                    f"train_acc={batch_record['running_train_acc']:.4f} "
                    f"batch_loss={batch_record['batch_loss']:.4f} "
                    f"batch_acc={batch_record['batch_acc']:.4f} "
                    f"lr={lr_before:.6g}",
                    flush=True,
                )

        final_train = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
            method=method,
            epoch=epoch,
            total_epochs=total_epochs,
            batch_callback=batch_callback if log_batches else None,
            log_interval=log_interval,
        )
        if scheduler is not None:
            scheduler.step()

        should_eval = (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1
        if should_eval:
            final_val = evaluate(
                model,
                val_loader,
                criterion,
                device,
                max_batches=max_val_batches,
                method=method,
            )
            if final_val["accuracy"] > best_val_accuracy:
                best_val_accuracy = final_val["accuracy"]
                best_epoch = epoch + 1

        epoch_record: dict[str, Any] = {
            **common_metrics,
            "epoch": epoch + 1,
            "epochs": total_epochs,
            "lr": lr_before,
            "train_loss": final_train["loss"],
            "train_acc": final_train["accuracy"],
            "val_loss": final_val["loss"],
            "val_acc": final_val["accuracy"],
            "duration_sec": time.perf_counter() - started_at,
            **_method_metrics(method, epoch=epoch, total_epochs=total_epochs),
        }
        if extra_epoch_metrics is not None:
            epoch_record.update(dict(extra_epoch_metrics(epoch, total_epochs)))
        if logger is not None:
            logger.log_epoch(epoch_record)
            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                logger.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    },
                    f"epoch_{epoch + 1:04d}.pt",
                )
        if log_to_console:
            print(
                "Epoch "
                f"[{epoch + 1}/{total_epochs}] "
                f"lr={lr_before:.6g} "
                f"train_loss={final_train['loss']:.4f} "
                f"train_acc={final_train['accuracy']:.4f} "
                f"test_loss={final_val['loss']:.4f} "
                f"test_acc={final_val['accuracy']:.4f} "
                f"model={common_metrics.get('model', '')} "
                f"dataset={common_metrics.get('dataset', '')} "
                f"scorer={common_metrics.get('scorer', '')} "
                f"sparsity={common_metrics.get('actual_sparsity', common_metrics.get('target_sparsity', ''))}",
                flush=True,
            )

    if best_val_accuracy == float("-inf"):
        best_val_accuracy = final_val["accuracy"]
    return FitResult(
        epochs=total_epochs,
        final_train=final_train,
        final_val=final_val,
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
    )
