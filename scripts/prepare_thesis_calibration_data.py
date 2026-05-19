"""Prepare public distilled calibration tensors used by thesis Chapter 4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.datasets import derive_ipc_subset, download_mtt_distilled


THESIS_DATASETS = ("cifar10", "cifar100", "tiny_imagenet")
DIRECT_MTT_IPCS = (1, 10)
DERIVED_IPCS = (5,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Chapter 4 public distilled calibration tensors.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/calibration"), help="Calibration data root.")
    parser.add_argument("--force", action="store_true", help="Re-download public tensors if present.")
    args = parser.parse_args()

    prepared: list[Path] = []
    for dataset in THESIS_DATASETS:
        for ipc in DIRECT_MTT_IPCS:
            root = download_mtt_distilled(dataset=dataset, ipc=ipc, output_dir=args.output_dir, force=args.force)
            prepared.append(root)
            print(f"downloaded_or_present: dataset={dataset} ipc={ipc} path={root}")

        source_root = args.output_dir / "mtt" / dataset / "ipc10"
        for derived_ipc in DERIVED_IPCS:
            target_root = args.output_dir / "mtt" / dataset / f"ipc{derived_ipc}"
            root = derive_ipc_subset(source_root=source_root, target_root=target_root, ipc=derived_ipc)
            prepared.append(root)
            print(f"derived: dataset={dataset} ipc={derived_ipc} path={root}")

    print("prepared_count:", len(prepared))


if __name__ == "__main__":
    main()
