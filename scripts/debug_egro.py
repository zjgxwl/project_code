"""Run a minimal TCSM -> EGRO Stage-1 debugging smoke test."""

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
from sso.methods import EGROMethod, TCSMMethod
from sso.models import build_model
from sso.pruning import build_score_dict
from sso.training import resolve_device, set_seed


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    parser = argparse.ArgumentParser(description="Run minimal EGRO Stage-1 group selection.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--scorer", type=str, default=None, help="Base pruning scorer to use.")
    parser.add_argument("--sparsity", type=float, default=None, help="TCSM target sparsity.")
    parser.add_argument("--flops-reduction", type=float, default=None, help="Target Conv FLOPs reduction.")
    args = parser.parse_args()

    config = load_config(args.config)
    scorer = resolve_scorer(config, args.scorer)
    sparsity = resolve_sparsity(config, args.sparsity)
    flops_reduction = resolve_flops_reduction(config, args.flops_reduction)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    train_loader, _ = build_dataloaders(config)
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
    metrics = egro_output.metrics

    print(f"device: {device}")
    print(f"scorer: {scorer}")
    print(f"target_flops_reduction: {metrics['target_flops_reduction']:.6f}")
    print(f"actual_flops_reduction: {metrics['flops_reduction']:.6f}")
    print(f"total_groups: {int(metrics['total_groups'])}")
    print(f"kept_groups: {int(metrics['kept_groups'])}")
    print(f"dropped_groups: {int(metrics['dropped_groups'])}")
    print(f"total_conv_params: {int(metrics['total_conv_params'])}")
    print(f"kept_conv_params: {int(metrics['kept_conv_params'])}")
    print(f"param_reduction: {metrics['param_reduction']:.6f}")
    print(f"total_conv_flops: {metrics['total_conv_flops']:.2f}")
    print(f"kept_conv_flops: {metrics['kept_conv_flops']:.2f}")


if __name__ == "__main__":
    main()
