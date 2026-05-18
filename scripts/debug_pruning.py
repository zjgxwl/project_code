"""Run a minimal pruning infrastructure smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.models import build_model
from sso.pruning import compute_sparsity, global_topk_mask, magnitude_score


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_mask_params(mask_dict: dict[str, Any]) -> tuple[int, int]:
    total_params = sum(mask.numel() for mask in mask_dict.values())
    kept_params = sum(int(mask.count_nonzero().item()) for mask in mask_dict.values())
    return total_params, kept_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug global magnitude pruning masks.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Target pruning sparsity.")
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config)
    scores = magnitude_score(model)
    masks = global_topk_mask(scores, sparsity=args.sparsity)
    total_params, kept_params = count_mask_params(masks)

    print(f"total_params: {total_params}")
    print(f"kept_params: {kept_params}")
    print(f"sparsity: {compute_sparsity(masks):.6f}")


if __name__ == "__main__":
    main()
