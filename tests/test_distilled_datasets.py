from pathlib import Path

import pytest
import torch

from sso.datasets import (
    DistilledTensorDataset,
    build_calibration_dataloader,
    build_distilled_calibration_loader,
    derive_ipc_subset,
    resolve_mtt_spec,
)


def _write_distilled_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(4, 3, 32, 32), root / "images_best.pt")
    torch.save(torch.tensor([0, 1, 2, 3]), root / "labels_best.pt")


def test_resolve_mtt_spec_supports_representative_cifar_sets() -> None:
    spec = resolve_mtt_spec("cifar10", 1)

    assert spec.folder == "cifar10"
    assert spec.num_classes == 10
    assert spec.image_size == 32
    tiny_spec = resolve_mtt_spec("tiny-imagenet", 10)
    assert tiny_spec.folder == "tiny_10"
    assert tiny_spec.num_classes == 200
    assert tiny_spec.image_size == 64


def test_distilled_tensor_dataset_loads_pt_tensors(tmp_path: Path) -> None:
    root = tmp_path / "mtt" / "cifar10" / "ipc1"
    _write_distilled_root(root)

    dataset = DistilledTensorDataset(root)
    image, label = dataset[1]

    assert len(dataset) == 4
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label.item() == 1


def test_build_calibration_dataloader_uses_distilled_source(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    _write_distilled_root(root)
    default_loader = build_distilled_calibration_loader(root, batch_size=2)
    config = {
        "dataset": {"batch_size": 2, "num_workers": 0},
        "calibration": {"source": "mtt", "data_dir": str(root), "batch_size": 2},
    }

    loader = build_calibration_dataloader(config, default_loader)
    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 3, 32, 32)
    assert targets.shape == (2,)


def test_build_calibration_dataloader_requires_distilled_path() -> None:
    config = {"calibration": {"source": "mtt"}}

    with pytest.raises(ValueError, match="calibration.data_dir"):
        build_calibration_dataloader(config, default_loader=None)  # type: ignore[arg-type]


def test_derive_ipc_subset_takes_first_samples_per_class(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(parents=True)
    images = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    torch.save(images, source / "images_best.pt")
    torch.save(labels, source / "labels_best.pt")

    derive_ipc_subset(source, target, ipc=2)
    derived = DistilledTensorDataset(target)

    assert len(derived) == 4
    derived_labels = [int(derived[index][1].item()) for index in range(len(derived))]
    assert derived_labels == [0, 0, 1, 1]
