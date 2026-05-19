"""Run a minimal pruning infrastructure smoke test."""

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
from sso.pruning import build_score_dict, compute_sparsity, global_topk_mask
from sso.training import resolve_device, set_seed
from sso.utils import add_data_args, apply_data_overrides, build_run_name, dataset_record_fields, get_timestamp, save_metrics


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_mask_params(mask_dict: dict[str, Any]) -> tuple[int, int]:
    total_params = sum(mask.numel() for mask in mask_dict.values())
    kept_params = sum(int(mask.count_nonzero().item()) for mask in mask_dict.values())
    return total_params, kept_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug global pruning masks.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Target pruning sparsity.")
    parser.add_argument("--scorer", type=str, default=None, help="Pruning scorer to use.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"), help="Directory for saved metrics.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional metrics run name.")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics JSON/JSONL.")
    add_data_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_data_overrides(config, args)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    scorer = (args.scorer or config.get("pruning", {}).get("scorer", "magnitude")).lower()

    train_loader = None
    criterion = None
    if scorer in {"snip", "grasp"}:
        train_loader, _ = build_dataloaders(config)
        criterion = nn.CrossEntropyLoss()

    model = build_model(config).to(device)
    scores = build_score_dict(
        model,
        config,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        scorer=scorer,
    )
    masks = global_topk_mask(scores, sparsity=args.sparsity)
    total_params, kept_params = count_mask_params(masks)

    print(f"scorer: {scorer}")
    print(f"total_params: {total_params}")
    print(f"kept_params: {kept_params}")
    print(f"sparsity: {compute_sparsity(masks):.6f}")
    if args.save_metrics:
        run_name = args.run_name or build_run_name(
            method="debug_pruning",
            scorer=scorer,
            model=config.get("model", {}).get("name", "resnet18"),
            sparsity=args.sparsity,
        )
        metrics_path = save_metrics(
            {
                "script": "debug_pruning.py",
                "method": "debug_pruning",
                "model": config.get("model", {}).get("name", "resnet18"),
                "scorer": scorer,
                "device": str(device),
                "seed": int(config.get("seed", 42)),
                "use_fake_data": bool(config.get("dataset", {}).get("use_fake_data", config.get("use_fake_data", True))),
                "fast_dev_run": bool(config.get("fast_dev_run", False)),
                "timestamp": get_timestamp(),
                **dataset_record_fields(config),
                "sparsity": compute_sparsity(masks),
                "total_params": total_params,
                "kept_params": kept_params,
            },
            output_dir=args.output_dir,
            run_name=run_name,
        )
        print(f"metrics_saved: {metrics_path}")


if __name__ == "__main__":
    main()
