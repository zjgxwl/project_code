"""Run EGRO Stage-3a VGG-style slim export validation."""

from __future__ import annotations

import argparse
import copy
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
from sso.export import export_vgg_slim_artifact
from sso.methods import EGROMethod, EGROTrainingMethod, TCSMMethod
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


def resolve_flops_reduction(config: dict[str, Any], cli_flops_reduction: float | None) -> float:
    if cli_flops_reduction is not None:
        return cli_flops_reduction
    return float(config.get("egro", {}).get("flops_reduction", 0.5))


def resolve_lambda0(config: dict[str, Any], cli_lambda0: float | None) -> float:
    if cli_lambda0 is not None:
        return cli_lambda0
    return float(config.get("egro", {}).get("lambda0", 1e-4))


def config_with_model(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("model", {})
    updated["model"]["name"] = model_name
    return updated


def build_tcsm_method(
    model: nn.Module,
    config: dict[str, Any],
    train_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    base_score_dict: dict[str, torch.Tensor],
    sparsity: float,
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
    )


def build_egro_method(
    model: nn.Module,
    config: dict[str, Any],
    stable_score_dict: dict[str, torch.Tensor],
    flops_reduction: float,
) -> EGROMethod:
    egro_config = config.get("egro", {})
    input_shape = egro_config.get("input_shape")
    if input_shape is None:
        image_size = int(config.get("dataset", {}).get("image_size", 32))
        input_shape = [1, 3, image_size, image_size]
    return EGROMethod(
        model=model,
        stable_score_dict=stable_score_dict,
        input_shape=input_shape,
        flops_reduction=flops_reduction,
        min_keep_ratio=float(egro_config.get("min_keep_ratio", 0.2)),
        safety_beta=float(egro_config.get("safety_beta", 0.0)),
        eta_g=float(egro_config.get("eta_g", 0.05)),
        delta=float(egro_config.get("delta", 1e-12)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal EGRO Stage-3a VGG slim export.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--model", type=str, default="vgg11_bn", help="Model name; only vgg11_bn is supported.")
    parser.add_argument("--scorer", type=str, default=None, help="Base pruning scorer to use.")
    parser.add_argument("--sparsity", type=float, default=None, help="TCSM target sparsity.")
    parser.add_argument("--flops-reduction", type=float, default=None, help="Target Conv FLOPs reduction.")
    parser.add_argument("--lambda0", type=float, default=None, help="Initial EGRO group regularization strength.")
    parser.add_argument("--skip-train", action="store_true", help="Export directly from Stage-1 group mask.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    add_data_args(parser)
    args = parser.parse_args()

    model_name = args.model.lower()
    if model_name != "vgg11_bn":
        raise ValueError("EGRO Stage-3a slim export only supports --model vgg11_bn.")

    base_config = load_config(args.config)
    config = config_with_model(base_config, model_name)
    apply_data_overrides(config, args)
    scorer = resolve_scorer(config, args.scorer)
    sparsity = resolve_sparsity(config, args.sparsity)
    flops_reduction = resolve_flops_reduction(config, args.flops_reduction)
    lambda0 = resolve_lambda0(config, args.lambda0)
    egro_config = config.get("egro", {})
    image_size = int(config.get("dataset", {}).get("image_size", 32))
    input_shape = egro_config.get("input_shape", [1, 3, image_size, image_size])
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = build_dataloaders(config)
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
        train_loader,
        criterion,
        device,
        base_score_dict,
        sparsity,
    )
    tcsm_output = tcsm.run()
    egro = build_egro_method(model, config, tcsm_output.stable_score_dict, flops_reduction)
    egro_output = egro.run()
    method = EGROTrainingMethod(
        model=model,
        groups=egro_output.groups,
        group_mask=egro_output.group_mask,
        group_omega=egro_output.group_omega,
        lambda0=lambda0,
        delta=float(egro_config.get("delta", 1e-12)),
    )

    fast_dev_run = bool(config.get("fast_dev_run", False))
    max_train_batches = int(config.get("max_train_batches", 2)) if fast_dev_run else None
    max_val_batches = int(config.get("max_val_batches", 2)) if fast_dev_run else None
    total_epochs = int(config.get("training", {}).get("epochs", 1))
    if not args.skip_train:
        optimizer = build_optimizer(model, config)
        train_one_epoch(
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

    export_artifact = export_vgg_slim_artifact(
        model=model,
        group_mask=egro_output.group_mask,
        groups=egro_output.groups,
        num_classes=int(config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 10))),
        input_shape=input_shape,
    )
    slim_model = export_artifact.model
    export_metadata = export_artifact.metadata
    slim_metrics = evaluate(
        slim_model,
        val_loader,
        criterion,
        device,
        max_batches=max_val_batches,
    )
    metrics = egro_output.metrics

    print(f"device: {device}")
    print(f"model: {model_name}")
    print(f"scorer: {scorer}")
    print(f"target_flops_reduction: {metrics['target_flops_reduction']:.6f}")
    print(f"actual_flops_reduction_stage1: {metrics['flops_reduction']:.6f}")
    print(f"original_params: {int(export_metadata['original_params'])}")
    print(f"slim_params: {int(export_metadata['slim_params'])}")
    print(f"param_reduction_actual: {export_metadata['param_reduction']:.6f}")
    print(f"original_conv_flops: {export_metadata['original_conv_flops']:.2f}")
    print(f"slim_conv_flops: {export_metadata['slim_conv_flops']:.2f}")
    print(f"flops_reduction_actual: {export_metadata['flops_reduction']:.6f}")
    print(f"mean_layer_keep_ratio: {metrics['mean_layer_keep_ratio']:.6f}")
    print(f"min_layer_keep_ratio: {metrics['min_layer_keep_ratio']:.6f}")
    print(f"slim_val_loss: {slim_metrics['loss']:.4f}")
    print(f"slim_val_acc: {slim_metrics['accuracy']:.4f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="egro_vgg_export",
            scorer=scorer,
            model=model_name,
            sparsity=sparsity,
            flops_reduction=flops_reduction,
        )
        metrics_path = save_metrics(
            {
                "script": "export_egro_vgg.py",
                "method": "egro_stage3a",
                "model": model_name,
                "scorer": scorer,
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                **dataset_record_fields(config),
                "sparsity": sparsity,
                "target_flops_reduction": metrics["target_flops_reduction"],
                "actual_flops_reduction_stage1": metrics["flops_reduction"],
                "total_groups": metrics["total_groups"],
                "kept_groups": metrics["kept_groups"],
                "dropped_groups": metrics["dropped_groups"],
                "group_param_reduction": metrics["param_reduction"],
                "total_conv_flops": metrics["total_conv_flops"],
                "kept_conv_flops": metrics["kept_conv_flops"],
                "mean_layer_keep_ratio": metrics["mean_layer_keep_ratio"],
                "min_layer_keep_ratio": metrics["min_layer_keep_ratio"],
                "layer_keep_ratios": egro_output.layer_keep_ratios,
                "lambda0": lambda0,
                "original_params": export_metadata["original_params"],
                "slim_params": export_metadata["slim_params"],
                "param_reduction_actual": export_metadata["param_reduction"],
                "original_conv_flops_actual": export_metadata["original_conv_flops"],
                "slim_conv_flops_actual": export_metadata["slim_conv_flops"],
                "flops_reduction_actual": export_metadata["flops_reduction"],
                "export_metadata": export_metadata,
                "slim_val_loss": slim_metrics["loss"],
                "slim_val_acc": slim_metrics["accuracy"],
            },
            output_dir=args.output_dir,
            run_name=run_name,
        )
        print(f"metrics_saved: {metrics_path}")


if __name__ == "__main__":
    main()
