"""Dataset builders for real dataset training and validation runs."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def normalize_dataset_name(name: str) -> str:
    """Normalize supported dataset aliases."""
    normalized = name.lower().replace("-", "_")
    if normalized == "tinyimagenet":
        return "tiny_imagenet"
    return normalized


def expected_num_classes(dataset_name: str) -> int:
    """Return the expected class count for supported datasets."""
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "tiny_imagenet":
        return 200
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _validate_num_classes(dataset_config: dict[str, Any], dataset_name: str) -> int:
    expected = expected_num_classes(dataset_name)
    configured = int(dataset_config.get("num_classes", expected))
    if configured != expected:
        raise ValueError(
            f"dataset.num_classes for {dataset_name} must be {expected}, got {configured}."
        )
    return configured


def _default_image_size(dataset_name: str) -> int:
    if dataset_name == "tiny_imagenet":
        return 64
    return 32


def _build_transform(dataset_name: str, image_size: int) -> transforms.Compose:
    if dataset_name == "tiny_imagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _apply_subset(dataset: Dataset, subset_size: int | None) -> Dataset:
    if subset_size is None:
        return dataset
    bounded_size = max(0, min(int(subset_size), len(dataset)))
    return Subset(dataset, list(range(bounded_size)))


def _get_subset_size(dataset_config: dict[str, Any], key: str) -> int | None:
    value = dataset_config.get(key)
    if value is None:
        return None
    return int(value)


class TinyImageNet200(Dataset):
    """Tiny-ImageNet-200 train/val reader without rewriting the val directory."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.wnids = self._read_wnids()
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples = self._build_samples()

    def _read_wnids(self) -> list[str]:
        wnids_path = self.root / "wnids.txt"
        if wnids_path.exists():
            return [
                line.strip()
                for line in wnids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        train_dir = self.root / "train"
        return sorted(path.name for path in train_dir.iterdir() if path.is_dir())

    def _build_samples(self) -> list[tuple[Path, int]]:
        if self.split == "train":
            samples: list[tuple[Path, int]] = []
            for wnid in self.wnids:
                image_dir = self.root / "train" / wnid / "images"
                if not image_dir.exists():
                    raise RuntimeError(f"Tiny-ImageNet train directory is missing: {image_dir}")
                for image_path in sorted(image_dir.iterdir()):
                    if image_path.is_file():
                        samples.append((image_path, self.class_to_idx[wnid]))
            return samples

        if self.split == "val":
            annotation_path = self.root / "val" / "val_annotations.txt"
            image_dir = self.root / "val" / "images"
            if not annotation_path.exists():
                raise RuntimeError(f"Tiny-ImageNet val annotations are missing: {annotation_path}")
            if not image_dir.exists():
                raise RuntimeError(f"Tiny-ImageNet val image directory is missing: {image_dir}")
            samples = []
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    continue
                image_path = image_dir / filename
                if image_path.exists():
                    samples.append((image_path, self.class_to_idx[wnid]))
            return samples

        raise ValueError(f"Unsupported Tiny-ImageNet split: {self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _tiny_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / "tiny-imagenet-200"


def _tiny_structure_complete(root: Path) -> bool:
    return (
        (root / "train").is_dir()
        and (root / "val" / "images").is_dir()
        and (root / "val" / "val_annotations.txt").is_file()
    )


def _extract_tiny_archive(archive_path: Path, data_dir: Path) -> None:
    if not archive_path.exists():
        raise RuntimeError(f"Tiny-ImageNet archive_path does not exist: {archive_path}")
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid Tiny-ImageNet zip archive: {archive_path}") from exc


def _download_tiny_archive(url: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "tiny-imagenet-200.zip"
    urlretrieve(url, archive_path)
    return archive_path


def _prepare_tiny_imagenet(dataset_config: dict[str, Any]) -> Path:
    data_dir = Path(dataset_config.get("data_dir", "data"))
    root = _tiny_root(data_dir)
    if _tiny_structure_complete(root):
        return root

    archive_value = dataset_config.get("archive_path")
    if archive_value:
        _extract_tiny_archive(Path(archive_value), data_dir)
        if _tiny_structure_complete(root):
            return root

    if bool(dataset_config.get("download", False)):
        url = str(dataset_config.get("url", TINY_IMAGENET_URL))
        archive_path = _download_tiny_archive(url, data_dir)
        _extract_tiny_archive(archive_path, data_dir)
        if _tiny_structure_complete(root):
            return root

    raise RuntimeError(
        "Tiny-ImageNet data was not found. Expected data_dir/tiny-imagenet-200 "
        "with train/ and val/val_annotations.txt. Provide dataset.archive_path, "
        "manually extract the archive, or pass --download."
    )


def _build_cifar_datasets(
    dataset_name: str,
    dataset_config: dict[str, Any],
    transform: transforms.Compose,
) -> tuple[Dataset, Dataset]:
    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    data_dir = dataset_config.get("data_dir", "data")
    download = bool(dataset_config.get("download", False))
    try:
        train_dataset = dataset_cls(root=data_dir, train=True, download=download, transform=transform)
        val_dataset = dataset_cls(root=data_dir, train=False, download=download, transform=transform)
    except RuntimeError as exc:
        if not download:
            raise RuntimeError(
                f"{dataset_name} data was not found in {data_dir}. "
                "Pass --download to download it or check dataset.data_dir."
            ) from exc
        raise
    return train_dataset, val_dataset


def _build_tiny_datasets(
    dataset_config: dict[str, Any],
    transform: transforms.Compose,
) -> tuple[Dataset, Dataset]:
    root = _prepare_tiny_imagenet(dataset_config)
    train_dataset = TinyImageNet200(root=root, split="train", transform=transform)
    val_dataset = TinyImageNet200(root=root, split="val", transform=transform)
    return train_dataset, val_dataset


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from real datasets."""
    dataset_config = config.get("dataset", {})
    dataset_name = normalize_dataset_name(dataset_config.get("name", "cifar10"))
    if dataset_name not in {"cifar10", "cifar100", "tiny_imagenet"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = _validate_num_classes(dataset_config, dataset_name)
    image_size = int(dataset_config.get("image_size", _default_image_size(dataset_name)))
    batch_size = int(dataset_config.get("batch_size", 8))
    num_workers = int(dataset_config.get("num_workers", config.get("num_workers", 0)))
    transform = _build_transform(dataset_name, image_size)

    if dataset_name in {"cifar10", "cifar100"}:
        train_dataset, val_dataset = _build_cifar_datasets(dataset_name, dataset_config, transform)
    else:
        train_dataset, val_dataset = _build_tiny_datasets(dataset_config, transform)

    train_dataset = _apply_subset(train_dataset, _get_subset_size(dataset_config, "train_subset_size"))
    val_dataset = _apply_subset(val_dataset, _get_subset_size(dataset_config, "val_subset_size"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
