"""Configuration helpers shared by experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sso.datasets.cifar import normalize_dataset_name


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add common dataset override arguments to a script parser."""
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset.data_dir.")
    parser.add_argument("--real-data", action="store_true", help="Use real dataset instead of FakeData.")
    parser.add_argument("--download", action="store_true", help="Allow explicit dataset download.")


def apply_data_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply common dataset CLI overrides in place and return config."""
    dataset_config = config.setdefault("dataset", {})
    if getattr(args, "real_data", False):
        dataset_config["use_fake_data"] = False
        config["use_fake_data"] = False
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
        "dataset_use_fake_data": bool(dataset_config.get("use_fake_data", config.get("use_fake_data", True))),
        "download": bool(dataset_config.get("download", False)),
        "train_subset_size": dataset_config.get("train_subset_size"),
        "val_subset_size": dataset_config.get("val_subset_size"),
    }
