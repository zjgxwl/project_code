"""Run a minimal TSPR training smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.datasets import build_dataloaders
from sso.methods import TSPRMethod
from sso.models import build_model
from sso.pruning import compute_sparsity, global_topk_mask, magnitude_score
from sso.training import evaluate, resolve_device, set_seed, train_one_epoch


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config.get("optimizer", {})
    name = optimizer_config.get("name", "sgd").lower()
    lr = float(optimizer_config.get("lr", 0.01))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def resolve_sparsity(config: dict[str, Any], cli_sparsity: float | None) -> float:
    if cli_sparsity is not None:
        return cli_sparsity
    return float(config.get("pruning", {}).get("sparsity", 0.9))


def resolve_lambda0(config: dict[str, Any], cli_lambda0: float | None) -> float:
    if cli_lambda0 is not None:
        return cli_lambda0
    return float(config.get("method", {}).get("lambda0", 1e-4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal TSPR training.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--sparsity", type=float, default=None, help="Target pruning sparsity.")
    parser.add_argument("--lambda0", type=float, default=None, help="Initial TSPR regularization strength.")
    args = parser.parse_args()

    config = load_config(args.config)
    pruning_config = config.get("pruning", {})
    scorer = pruning_config.get("scorer", "magnitude").lower()
    if scorer != "magnitude":
        raise ValueError(f"Unsupported TSPR scorer: {scorer}")

    sparsity = resolve_sparsity(config, args.sparsity)
    method_config = config.get("method", {})
    lambda0 = resolve_lambda0(config, args.lambda0)
    eps = float(method_config.get("eps", 1e-8))
    delta = float(method_config.get("delta", 1e-12))

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    score_dict = magnitude_score(model)
    mask_dict = global_topk_mask(score_dict, sparsity=sparsity)
    method = TSPRMethod(
        model,
        mask_dict,
        score_dict,
        lambda0=lambda0,
        eps=eps,
        delta=delta,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    fast_dev_run = bool(config.get("fast_dev_run", False))
    max_train_batches = int(config.get("max_train_batches", 2)) if fast_dev_run else None
    max_val_batches = int(config.get("max_val_batches", 2)) if fast_dev_run else None
    total_epochs = int(config.get("training", {}).get("epochs", 1))

    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        max_batches=max_train_batches,
        method=method,
        epoch=0,
        total_epochs=total_epochs,
    )

    eval_model = build_model(config).to(device)
    eval_model.load_state_dict(method.export_state_dict())
    val_metrics = evaluate(
        eval_model,
        val_loader,
        criterion,
        device,
        max_batches=max_val_batches,
    )

    print(f"device: {device}")
    print(f"sparsity: {compute_sparsity(method.fixed_mask_dict):.6f}")
    print(f"lambda0: {lambda0:g}")
    print(f"train_loss: {train_metrics['loss']:.4f}")
    print(f"train_acc: {train_metrics['accuracy']:.4f}")
    print(f"val_loss: {val_metrics['loss']:.4f}")
    print(f"val_acc: {val_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
