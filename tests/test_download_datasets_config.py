import argparse
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from sso.datasets import build_dataloaders
from sso.datasets.cifar import normalize_dataset_name
from sso.utils.config import apply_data_overrides, dataset_record_fields


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    with (PROJECT_ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_image(path: Path, color: tuple[int, int, int] = (128, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _make_tiny_root(root: Path) -> None:
    wnids = ["n00000001", "n00000002"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "wnids.txt").write_text("\n".join(wnids) + "\n", encoding="utf-8")
    for index, wnid in enumerate(wnids):
        _write_image(root / "train" / wnid / "images" / f"{wnid}_0.JPEG", color=(index * 20, 64, 32))
    _write_image(root / "val" / "images" / "val_0.JPEG")
    (root / "val" / "val_annotations.txt").write_text(
        "val_0.JPEG\tn00000002\t0\t0\t8\t8\n",
        encoding="utf-8",
    )


def test_download_smoke_configs_load_and_are_opt_in() -> None:
    for name, classes in (
        ("cifar10_download_smoke.yaml", 10),
        ("cifar100_download_smoke.yaml", 100),
        ("tiny_imagenet_download_smoke.yaml", 200),
    ):
        config = _load_config(name)
        dataset = config["dataset"]
        assert dataset["download"] is False
        assert dataset["use_fake_data"] is False
        assert dataset["train_subset_size"] == 128
        assert dataset["val_subset_size"] == 64
        assert dataset["num_classes"] == classes
        assert config["model"]["num_classes"] == classes


def test_fake_data_supports_tiny_imagenet_shape_and_classes() -> None:
    config = {
        "dataset": {
            "name": "tinyimagenet",
            "use_fake_data": True,
            "num_classes": 200,
            "image_size": 64,
            "batch_size": 2,
            "train_subset_size": 3,
            "val_subset_size": 2,
            "num_workers": 0,
        }
    }

    train_loader, val_loader = build_dataloaders(config)

    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (2, 3, 64, 64)
    assert targets.max().item() < 200
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 2


def test_real_cifar_without_download_and_empty_dir_errors(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "name": "cifar10",
            "use_fake_data": False,
            "download": False,
            "data_dir": str(tmp_path),
            "num_classes": 10,
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
        }
    }

    with pytest.raises(RuntimeError, match="Pass --download"):
        build_dataloaders(config)


def test_tiny_imagenet_without_data_or_download_errors(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "name": "tiny-imagenet",
            "use_fake_data": False,
            "download": False,
            "data_dir": str(tmp_path),
            "num_classes": 200,
            "image_size": 64,
            "batch_size": 2,
            "num_workers": 0,
        }
    }

    with pytest.raises(RuntimeError, match="Tiny-ImageNet data was not found"):
        build_dataloaders(config)


def test_tiny_imagenet_minimal_directory_reads_train_and_val(tmp_path: Path) -> None:
    tiny_root = tmp_path / "tiny-imagenet-200"
    _make_tiny_root(tiny_root)
    config = {
        "dataset": {
            "name": "tiny_imagenet",
            "use_fake_data": False,
            "download": False,
            "data_dir": str(tmp_path),
            "num_classes": 200,
            "image_size": 64,
            "batch_size": 1,
            "train_subset_size": 1,
            "val_subset_size": 1,
            "num_workers": 0,
        }
    }

    train_loader, val_loader = build_dataloaders(config)

    train_inputs, train_targets = next(iter(train_loader))
    val_inputs, val_targets = next(iter(val_loader))
    assert train_inputs.shape == (1, 3, 64, 64)
    assert val_inputs.shape == (1, 3, 64, 64)
    assert train_targets.item() == 0
    assert val_targets.item() == 1


def test_tiny_imagenet_archive_path_extracts_before_download(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "tiny-imagenet-200"
    _make_tiny_root(source_root)
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path / "source"))

    data_dir = tmp_path / "data"
    config = {
        "dataset": {
            "name": "tiny_imagenet",
            "use_fake_data": False,
            "download": False,
            "archive_path": str(archive_path),
            "data_dir": str(data_dir),
            "num_classes": 200,
            "image_size": 64,
            "batch_size": 1,
            "num_workers": 0,
        }
    }

    train_loader, _ = build_dataloaders(config)

    assert (data_dir / "tiny-imagenet-200" / "val" / "val_annotations.txt").exists()
    assert len(train_loader.dataset) == 2


def test_dataset_name_aliases_and_num_class_validation() -> None:
    assert normalize_dataset_name("tiny-imagenet") == "tiny_imagenet"
    assert normalize_dataset_name("tiny_imagenet") == "tiny_imagenet"
    assert normalize_dataset_name("tinyimagenet") == "tiny_imagenet"

    config = {
        "dataset": {
            "name": "tinyimagenet",
            "use_fake_data": True,
            "num_classes": 10,
            "image_size": 64,
            "batch_size": 2,
        }
    }
    with pytest.raises(ValueError, match="num_classes"):
        build_dataloaders(config)


def test_data_cli_overrides_and_record_fields() -> None:
    config = {"dataset": {"name": "cifar10", "use_fake_data": True, "data_dir": "old"}}
    args = argparse.Namespace(real_data=True, download=True, data_dir=Path("new_data"))

    apply_data_overrides(config, args)
    fields = dataset_record_fields(config)

    assert config["dataset"]["use_fake_data"] is False
    assert config["dataset"]["download"] is True
    assert config["dataset"]["data_dir"] == "new_data"
    assert fields["dataset"] == "cifar10"
    assert fields["data_dir"] == "new_data"
    assert fields["dataset_use_fake_data"] is False
    assert fields["download"] is True
