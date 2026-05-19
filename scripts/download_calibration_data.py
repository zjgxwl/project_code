"""Download public distilled calibration tensors for TCSM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.datasets import download_mtt_distilled


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public distilled calibration data.")
    parser.add_argument("--source", type=str, default="mtt", choices=["mtt"], help="Public distilled dataset source.")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name, e.g. cifar10 or cifar100.")
    parser.add_argument("--ipc", type=int, default=1, help="Images per class.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration"), help="Calibration data root.")
    parser.add_argument("--force", action="store_true", help="Re-download files if they already exist.")
    args = parser.parse_args()

    if args.source != "mtt":
        raise ValueError(f"Unsupported calibration source: {args.source}")
    root = download_mtt_distilled(
        dataset=args.dataset,
        ipc=args.ipc,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"calibration_source: {args.source}")
    print(f"dataset: {args.dataset}")
    print(f"ipc: {args.ipc}")
    print(f"calibration_data_dir: {root}")


if __name__ == "__main__":
    main()
