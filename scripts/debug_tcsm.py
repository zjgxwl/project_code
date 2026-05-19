"""Run a minimal TCSM debugging smoke test."""

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
from sso.methods import TCSMMethod
from sso.models import build_model
from sso.pruning import build_score_dict
from sso.training import resolve_device, set_seed
from sso.utils import build_run_name, get_timestamp, save_metrics


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal TCSM scoring.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--scorer", type=str, default=None, help="Base pruning scorer to use.")
    parser.add_argument("--sparsity", type=float, default=None, help="Target pruning sparsity.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    args = parser.parse_args()

    config = load_config(args.config)
    scorer = resolve_scorer(config, args.scorer)
    sparsity = resolve_sparsity(config, args.sparsity)
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
    output = tcsm.run()

    print(f"device: {device}")
    print(f"scorer: {scorer}")
    print(f"sparsity: {sparsity:.6f}")
    print(f"stable_sparsity: {output.metrics['sparsity']:.6f}")
    print(f"base_stable_jaccard: {output.metrics['base_stable_jaccard']:.6f}")
    print(f"mean_stable_score: {output.metrics['mean_stable_score']:.6f}")
    print(f"mean_stable_omega: {output.metrics['mean_stable_omega']:.6f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="tcsm_debug",
            scorer=scorer,
            model=config.get("model", {}).get("name", "resnet18"),
            sparsity=sparsity,
        )
        metrics_path = save_metrics(
            {
                "script": "debug_tcsm.py",
                "method": "tcsm",
                "model": config.get("model", {}).get("name", "resnet18"),
                "scorer": scorer,
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "use_fake_data": bool(config.get("dataset", {}).get("use_fake_data", config.get("use_fake_data", True))),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                "sparsity": sparsity,
                "stable_sparsity": output.metrics["sparsity"],
                "base_stable_jaccard": output.metrics["base_stable_jaccard"],
                "mean_stable_score": output.metrics["mean_stable_score"],
                "mean_stable_omega": output.metrics["mean_stable_omega"],
            },
            output_dir=args.output_dir,
            run_name=run_name,
        )
        print(f"metrics_saved: {metrics_path}")


if __name__ == "__main__":
    main()
