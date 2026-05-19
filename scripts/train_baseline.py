"""Run a minimal baseline training validation."""

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
from sso.models import build_model
from sso.training import evaluate, resolve_device, set_seed, train_one_epoch
from sso.utils import add_data_args, apply_data_overrides, build_run_name, dataset_record_fields, get_timestamp, save_metrics


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal baseline training validation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    add_data_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_data_overrides(config, args)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    fast_dev_run = bool(config.get("fast_dev_run", False))
    max_train_batches = int(config.get("max_train_batches", 2)) if fast_dev_run else None
    max_val_batches = int(config.get("max_val_batches", 2)) if fast_dev_run else None

    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        max_batches=max_train_batches,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        max_batches=max_val_batches,
    )

    print(f"device: {device}")
    print(f"train_loss: {train_metrics['loss']:.4f}")
    print(f"train_acc: {train_metrics['accuracy']:.4f}")
    print(f"val_loss: {val_metrics['loss']:.4f}")
    print(f"val_acc: {val_metrics['accuracy']:.4f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="baseline",
            model=config.get("model", {}).get("name", "resnet18"),
        )
        metrics_path = save_metrics(
            {
                "script": "train_baseline.py",
                "method": "baseline",
                "model": config.get("model", {}).get("name", "resnet18"),
                "scorer": None,
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                **dataset_record_fields(config),
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            },
            output_dir=args.output_dir,
            run_name=run_name,
        )
        print(f"metrics_saved: {metrics_path}")


if __name__ == "__main__":
    main()
