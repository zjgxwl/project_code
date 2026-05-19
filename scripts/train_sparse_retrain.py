"""Run a minimal fixed-mask sparse retraining smoke test."""

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
from sso.methods import StandardSparseRetrainingMethod
from sso.models import build_model
from sso.pruning import build_score_dict, compute_sparsity, global_topk_mask
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


def resolve_sparsity(config: dict[str, Any], cli_sparsity: float | None) -> float:
    if cli_sparsity is not None:
        return cli_sparsity
    return float(config.get("pruning", {}).get("sparsity", 0.9))


def resolve_scorer(config: dict[str, Any], cli_scorer: str | None) -> str:
    if cli_scorer is not None:
        return cli_scorer.lower()
    return str(config.get("pruning", {}).get("scorer", "magnitude")).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-mask sparse retraining.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--sparsity", type=float, default=None, help="Target pruning sparsity.")
    parser.add_argument("--scorer", type=str, default=None, help="Pruning scorer to use.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    add_data_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_data_overrides(config, args)
    scorer = resolve_scorer(config, args.scorer)
    sparsity = resolve_sparsity(config, args.sparsity)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    score_dict = build_score_dict(
        model,
        config,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        scorer=scorer,
    )
    mask_dict = global_topk_mask(score_dict, sparsity=sparsity)
    method = StandardSparseRetrainingMethod(model, mask_dict)

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
        method=method,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        max_batches=max_val_batches,
        method=method,
    )

    print(f"device: {device}")
    print(f"sparsity: {compute_sparsity(method.mask_state_dict()):.6f}")
    print(f"train_loss: {train_metrics['loss']:.4f}")
    print(f"train_acc: {train_metrics['accuracy']:.4f}")
    print(f"val_loss: {val_metrics['loss']:.4f}")
    print(f"val_acc: {val_metrics['accuracy']:.4f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="sparse_retrain",
            scorer=scorer,
            model=config.get("model", {}).get("name", "resnet18"),
            sparsity=sparsity,
        )
        total_params = sum(mask.numel() for mask in method.mask_state_dict().values())
        kept_params = sum(int(mask.count_nonzero().item()) for mask in method.mask_state_dict().values())
        metrics_path = save_metrics(
            {
                "script": "train_sparse_retrain.py",
                "method": "sparse_retrain",
                "model": config.get("model", {}).get("name", "resnet18"),
                "scorer": scorer,
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "use_fake_data": bool(config.get("dataset", {}).get("use_fake_data", config.get("use_fake_data", True))),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                **dataset_record_fields(config),
                "sparsity": compute_sparsity(method.mask_state_dict()),
                "total_params": total_params,
                "kept_params": kept_params,
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
