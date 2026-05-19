"""Run a minimal TCSM -> TSPR validation."""

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

from sso.datasets import build_calibration_dataloader, build_dataloaders, download_mtt_distilled
from sso.methods import TCSMMethod, TSPRMethod
from sso.models import build_model
from sso.pruning import build_score_dict
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


def resolve_scorer(config: dict[str, Any], cli_scorer: str | None) -> str:
    if cli_scorer is not None:
        return cli_scorer.lower()
    return str(config.get("pruning", {}).get("scorer", "magnitude")).lower()


def resolve_sparsity(config: dict[str, Any], cli_sparsity: float | None) -> float:
    if cli_sparsity is not None:
        return cli_sparsity
    return float(config.get("pruning", {}).get("sparsity", 0.9))


def resolve_lambda0(config: dict[str, Any], cli_lambda0: float | None) -> float:
    if cli_lambda0 is not None:
        return cli_lambda0
    return float(config.get("method", {}).get("lambda0", 1e-4))


def apply_calibration_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "calibration_source", "train") == "train":
        return
    calibration_config = config.setdefault("calibration", {})
    calibration_config["source"] = args.calibration_source
    calibration_config["ipc"] = int(args.calibration_ipc)
    if args.calibration_source == "mtt" and args.download_calibration:
        root = download_mtt_distilled(
            dataset=config.get("dataset", {}).get("name", "cifar10"),
            ipc=int(args.calibration_ipc),
            output_dir=args.calibration_output_dir,
            force=bool(args.force_calibration_download),
        )
        calibration_config["data_dir"] = str(root)
    elif args.calibration_data_dir is not None:
        calibration_config["data_dir"] = str(args.calibration_data_dir)


def build_tcsm_method(
    model: nn.Module,
    config: dict[str, Any],
    train_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    base_score_dict: dict[str, torch.Tensor],
    sparsity: float,
    calibration_mode: str | None,
    stability_repeats: int | None,
) -> TCSMMethod:
    tcsm_config = config.get("tcsm", {})
    return TCSMMethod(
        model=model,
        base_score_dict=base_score_dict,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        sparsity=sparsity,
        alpha=float(tcsm_config.get("alpha", 0.5)),
        beta=float(tcsm_config.get("beta", 0.5)),
        eta=float(tcsm_config.get("eta", 0.05)),
        delta=float(tcsm_config.get("delta", 1e-12)),
        calibration_batches=int(tcsm_config.get("calibration_batches", 1)),
        sign_batches=int(tcsm_config.get("sign_batches", 1)),
        calibration_mode=calibration_mode or str(tcsm_config.get("calibration_mode", "tcsm")),
        stability_repeats=(
            int(stability_repeats)
            if stability_repeats is not None
            else int(tcsm_config.get("stability_repeats", 1))
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal TCSM-stabilized TSPR.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--scorer", type=str, default=None, help="Base pruning scorer to use.")
    parser.add_argument("--sparsity", type=float, default=None, help="Target pruning sparsity.")
    parser.add_argument("--lambda0", type=float, default=None, help="Initial TSPR regularization strength.")
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default=None,
        choices=["base_only", "tcsm", "random_subset", "full_data"],
        help="TCSM calibration mode.",
    )
    parser.add_argument("--stability-repeats", type=int, default=None, help="Number of lightweight stability repeats.")
    parser.add_argument("--calibration-source", type=str, default="train", choices=["train", "mtt"], help="Calibration data source.")
    parser.add_argument("--calibration-ipc", type=int, default=1, help="Images per class for public distilled calibration data.")
    parser.add_argument("--calibration-data-dir", type=Path, default=None, help="Existing local distilled calibration directory.")
    parser.add_argument("--calibration-output-dir", type=Path, default=Path("data/calibration"), help="Download root for calibration data.")
    parser.add_argument("--download-calibration", action="store_true", help="Explicitly download public calibration data.")
    parser.add_argument("--force-calibration-download", action="store_true", help="Re-download calibration data if present.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    add_data_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_data_overrides(config, args)
    apply_calibration_overrides(config, args)
    scorer = resolve_scorer(config, args.scorer)
    sparsity = resolve_sparsity(config, args.sparsity)
    lambda0 = resolve_lambda0(config, args.lambda0)
    method_config = config.get("method", {})
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = build_dataloaders(config)
    calibration_loader = build_calibration_dataloader(config, train_loader)
    criterion = nn.CrossEntropyLoss()
    model = build_model(config).to(device)
    base_score_dict = build_score_dict(
        model,
        config,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        scorer=scorer,
    )
    tcsm = build_tcsm_method(
        model,
        config,
        calibration_loader,
        criterion,
        device,
        base_score_dict,
        sparsity,
        args.calibration_mode,
        args.stability_repeats,
    )
    tcsm_output = tcsm.run()
    method = TSPRMethod(
        model,
        mask_dict=tcsm_output.stable_mask_dict,
        score_dict=tcsm_output.stable_score_dict,
        lambda0=lambda0,
        eps=float(method_config.get("eps", 1e-8)),
        delta=float(method_config.get("delta", 1e-12)),
    )

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
    print(f"scorer: {scorer}")
    print(f"calibration_source: {config.get('calibration', {}).get('source', 'train')}")
    print(f"calibration_mode: {tcsm_output.metrics['calibration_mode']}")
    print(f"sparsity: {sparsity:.6f}")
    print(f"stable_sparsity: {tcsm_output.metrics['sparsity']:.6f}")
    print(f"base_stable_jaccard: {tcsm_output.metrics['base_stable_jaccard']:.6f}")
    print(f"score_rank_correlation: {tcsm_output.metrics['score_rank_correlation']:.6f}")
    print(f"mask_jaccard: {tcsm_output.metrics['mask_jaccard']:.6f}")
    print(f"lambda0: {lambda0:g}")
    print(f"train_loss: {train_metrics['loss']:.4f}")
    print(f"train_acc: {train_metrics['accuracy']:.4f}")
    print(f"val_loss: {val_metrics['loss']:.4f}")
    print(f"val_acc: {val_metrics['accuracy']:.4f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="tcsm_tspr",
            scorer=scorer,
            model=config.get("model", {}).get("name", "resnet18"),
            sparsity=sparsity,
        )
        metrics_path = save_metrics(
            {
                "script": "train_tcsm_tspr.py",
                "method": "tcsm_tspr",
                "model": config.get("model", {}).get("name", "resnet18"),
                "scorer": scorer,
                "calibration_source": config.get("calibration", {}).get("source", "train"),
                "calibration_data_dir": config.get("calibration", {}).get("data_dir"),
                "calibration_ipc": config.get("calibration", {}).get("ipc"),
                "calibration_mode": tcsm_output.metrics["calibration_mode"],
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                **dataset_record_fields(config),
                "sparsity": sparsity,
                "stable_sparsity": tcsm_output.metrics["sparsity"],
                "base_stable_jaccard": tcsm_output.metrics["base_stable_jaccard"],
                "score_rank_correlation": tcsm_output.metrics["score_rank_correlation"],
                "mask_jaccard": tcsm_output.metrics["mask_jaccard"],
                "stability_repeats": tcsm_output.metrics["stability_repeats"],
                "mean_stable_score": tcsm_output.metrics["mean_stable_score"],
                "mean_stable_omega": tcsm_output.metrics["mean_stable_omega"],
                "lambda0": lambda0,
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
