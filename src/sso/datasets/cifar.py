"""CIFAR-style dataloaders used by debug and smoke-test runs."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders.

    The debug path uses FakeData by default so tests and smoke runs do not
    download real datasets.
    """
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "cifar10").lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_classes = int(dataset_config.get("num_classes", 10 if dataset_name == "cifar10" else 100))
    image_size = int(dataset_config.get("image_size", 32))
    batch_size = int(dataset_config.get("batch_size", 8))
    num_workers = int(dataset_config.get("num_workers", config.get("num_workers", 0)))
    use_fake_data = bool(dataset_config.get("use_fake_data", config.get("use_fake_data", True)))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=max(batch_size * 4, 32),
            image_size=(3, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )
        val_dataset = datasets.FakeData(
            size=max(batch_size * 2, 16),
            image_size=(3, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )
    else:
        dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        data_dir = dataset_config.get("data_dir", "data")
        train_dataset = dataset_cls(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = dataset_cls(root=data_dir, train=False, download=True, transform=transform)

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
