"""Offline/public distilled calibration dataset helpers."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .cifar import normalize_dataset_name


MTT_BASE_URL = "https://georgecazenavette.github.io/mtt-distillation/tensors"


@dataclass(frozen=True)
class MTTDistilledSpec:
    """Download metadata for public MTT distilled tensor datasets."""

    dataset: str
    ipc: int
    folder: str
    num_classes: int
    image_size: int
    method: str = "mtt"


MTT_SPECS: dict[tuple[str, int], MTTDistilledSpec] = {
    ("cifar10", 1): MTTDistilledSpec("cifar10", 1, "cifar10", 10, 32),
    ("cifar10", 10): MTTDistilledSpec("cifar10", 10, "cifar10_10", 10, 32),
    ("cifar10", 50): MTTDistilledSpec("cifar10", 50, "cifar10_50", 10, 32),
    ("cifar100", 1): MTTDistilledSpec("cifar100", 1, "cifar100", 100, 32),
    ("cifar100", 10): MTTDistilledSpec("cifar100", 10, "cifar100_10", 100, 32),
    ("cifar100", 50): MTTDistilledSpec("cifar100", 50, "cifar100_50", 100, 32),
    ("tiny_imagenet", 1): MTTDistilledSpec("tiny_imagenet", 1, "tiny", 200, 64),
    ("tiny_imagenet", 10): MTTDistilledSpec("tiny_imagenet", 10, "tiny_10", 200, 64),
}


def resolve_mtt_spec(dataset: str, ipc: int) -> MTTDistilledSpec:
    """Resolve a supported public MTT tensor dataset."""
    normalized = normalize_dataset_name(dataset)
    key = (normalized, int(ipc))
    if key not in MTT_SPECS:
        supported = ", ".join(f"{name}:ipc{spec_ipc}" for name, spec_ipc in sorted(MTT_SPECS))
        raise ValueError(f"Unsupported MTT distilled dataset {dataset} ipc={ipc}. Supported: {supported}")
    return MTT_SPECS[key]


def _download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def download_mtt_distilled(
    dataset: str,
    ipc: int,
    output_dir: str | Path = "data/calibration",
    force: bool = False,
) -> Path:
    """Download public MTT distilled tensors and return their local directory."""
    spec = resolve_mtt_spec(dataset, ipc)
    root = Path(output_dir) / "mtt" / spec.dataset / f"ipc{spec.ipc}"
    images_url = f"{MTT_BASE_URL}/{spec.folder}/images_best.pt"
    labels_url = f"{MTT_BASE_URL}/{spec.folder}/labels_best.pt"
    _download_file(images_url, root / "images_best.pt", force=force)
    _download_file(labels_url, root / "labels_best.pt", force=force)

    metadata = {
        "source": "mtt",
        "dataset": spec.dataset,
        "ipc": spec.ipc,
        "num_classes": spec.num_classes,
        "image_size": spec.image_size,
        "folder": spec.folder,
        "images_url": images_url,
        "labels_url": labels_url,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def derive_ipc_subset(
    source_root: str | Path,
    target_root: str | Path,
    ipc: int,
    source: str = "mtt",
) -> Path:
    """Derive a smaller IPC tensor set by taking the first samples per class."""
    source_path = Path(source_root)
    target_path = Path(target_root)
    images = torch.load(source_path / "images_best.pt", map_location="cpu")
    labels = torch.load(source_path / "labels_best.pt", map_location="cpu").long().flatten()
    selected_indices: list[int] = []
    for label in sorted(int(value) for value in labels.unique().tolist()):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if int(indices.numel()) < ipc:
            raise ValueError(f"Class {label} has fewer than {ipc} samples in {source_path}.")
        selected_indices.extend(int(index) for index in indices[:ipc].tolist())
    selected = torch.tensor(selected_indices, dtype=torch.long)

    target_path.mkdir(parents=True, exist_ok=True)
    torch.save(images.index_select(0, selected), target_path / "images_best.pt")
    torch.save(labels.index_select(0, selected), target_path / "labels_best.pt")

    metadata: dict[str, Any] = {
        "source": source,
        "derived": True,
        "derived_from": str(source_path),
        "ipc": int(ipc),
        "num_classes": int(labels.unique().numel()),
        "num_samples": int(selected.numel()),
    }
    source_metadata = source_path / "metadata.json"
    if source_metadata.exists():
        metadata["source_metadata"] = json.loads(source_metadata.read_text(encoding="utf-8"))
    (target_path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target_path


class DistilledTensorDataset(Dataset):
    """Dataset backed by ``images_best.pt`` and ``labels_best.pt`` tensors."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        images_path = self.root / "images_best.pt"
        labels_path = self.root / "labels_best.pt"
        if not images_path.exists() or not labels_path.exists():
            raise RuntimeError(f"Distilled tensors not found under {self.root}.")
        self.images = self._normalize_image_tensor(torch.load(images_path, map_location="cpu"))
        self.labels = torch.load(labels_path, map_location="cpu").long().flatten()
        if int(self.images.shape[0]) != int(self.labels.shape[0]):
            raise ValueError("Distilled images and labels must have the same length.")

    def _normalize_image_tensor(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Distilled images must be 4D, got shape {tuple(images.shape)}.")
        if images.shape[1] not in {1, 3} and images.shape[-1] in {1, 3}:
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.shape[1] not in {1, 3}:
            raise ValueError(f"Expected channel-first distilled images, got shape {tuple(images.shape)}.")
        return images.float()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def build_distilled_calibration_loader(
    root: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a calibration dataloader from local distilled tensors."""
    dataset = DistilledTensorDataset(root)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(num_workers), pin_memory=False)


def build_calibration_dataloader(config: dict[str, Any], default_loader: DataLoader) -> DataLoader:
    """Return the configured calibration loader or fall back to the train loader."""
    calibration_config = config.get("calibration", {})
    source = str(calibration_config.get("source", "train")).lower()
    if source in {"train", "default"}:
        return default_loader
    if source in {"distilled", "mtt"}:
        root = calibration_config.get("data_dir")
        if root is None:
            raise ValueError("calibration.data_dir is required for distilled calibration data.")
        dataset_config = config.get("dataset", {})
        batch_size = int(calibration_config.get("batch_size", dataset_config.get("batch_size", 8)))
        num_workers = int(calibration_config.get("num_workers", dataset_config.get("num_workers", 0)))
        return build_distilled_calibration_loader(root, batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"Unsupported calibration source: {source}")
