"""Configuration helpers shared by experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sso.datasets.cifar import normalize_dataset_name


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add common dataset override arguments to a script parser."""
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset.data_dir.")
    parser.add_argument("--download", action="store_true", help="Allow explicit dataset download.")


def apply_data_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply common dataset CLI overrides in place and return config."""
    dataset_config = config.setdefault("dataset", {})
    if getattr(args, "data_dir", None) is not None:
        dataset_config["data_dir"] = str(args.data_dir)
    if getattr(args, "download", False):
        dataset_config["download"] = True
    return config


def dataset_record_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Return consistent dataset metadata for metric records."""
    dataset_config = config.get("dataset", {})
    dataset_name = normalize_dataset_name(str(dataset_config.get("name", "cifar10")))
    return {
        "dataset": dataset_name,
        "data_dir": dataset_config.get("data_dir", "data"),
        "download": bool(dataset_config.get("download", False)),
        "train_subset_size": dataset_config.get("train_subset_size"),
        "val_subset_size": dataset_config.get("val_subset_size"),
    }


def require_config_sections(
    config: dict[str, Any],
    sections: tuple[str, ...] = ("dataset", "model"),
) -> None:
    """Validate that required top-level config sections exist and are mappings."""
    missing = [section for section in sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    invalid = [section for section in sections if not isinstance(config.get(section), dict)]
    if invalid:
        raise ValueError(f"Config sections must be mappings: {invalid}")


def validate_experiment_config(
    config: dict[str, Any],
    required_sections: tuple[str, ...] = ("dataset", "model"),
) -> dict[str, Any]:
    """Validate common experiment config shape and return the same config."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")
    require_config_sections(config, required_sections)

    dataset_config = config.get("dataset", {})
    dataset_name = normalize_dataset_name(str(dataset_config.get("name", "cifar10")))
    if dataset_name not in {"cifar10", "cifar100", "tiny_imagenet"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "resnet18")).lower()
    if not model_name:
        raise ValueError("model.name must not be empty.")

    training_config = config.get("training", {})
    if training_config and int(training_config.get("epochs", 1)) < 0:
        raise ValueError("training.epochs must be non-negative.")

    pruning_config = config.get("pruning", {})
    if "sparsity" in pruning_config:
        sparsity = float(pruning_config["sparsity"])
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("pruning.sparsity must be in [0.0, 1.0).")

    egro_config = config.get("egro", {})
    if "flops_reduction" in egro_config:
        flops_reduction = float(egro_config["flops_reduction"])
        if not 0.0 <= flops_reduction <= 1.0:
            raise ValueError("egro.flops_reduction must be in [0.0, 1.0].")

    return config
