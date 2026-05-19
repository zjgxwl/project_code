import torch

from sso.datasets import build_dataloaders
from sso.models import build_model


def test_cifar_resnet18_forward_shape_with_real_data() -> None:
    batch_size = 4
    num_classes = 10
    config = {
        "dataset": {
            "name": "cifar10",
            "data_dir": "outputs/thesis_real_data",
            "download": False,
            "num_classes": num_classes,
            "image_size": 32,
            "batch_size": batch_size,
            "num_workers": 0,
            "train_subset_size": batch_size,
            "val_subset_size": batch_size,
        },
        "model": {
            "name": "resnet18",
            "num_classes": num_classes,
        },
    }

    train_loader, _ = build_dataloaders(config)
    model = build_model(config)
    inputs, _targets = next(iter(train_loader))

    with torch.no_grad():
        logits = model(inputs)

    assert list(logits.shape) == [batch_size, num_classes]


def test_thesis_cnn_models_forward_shape_with_real_data() -> None:
    batch_size = 2
    num_classes = 10
    config = {
        "dataset": {
            "name": "cifar10",
            "data_dir": "outputs/thesis_real_data",
            "download": False,
            "num_classes": num_classes,
            "image_size": 32,
            "batch_size": batch_size,
            "num_workers": 0,
            "train_subset_size": batch_size,
            "val_subset_size": batch_size,
        },
        "model": {
            "name": "resnet18",
            "num_classes": num_classes,
        },
    }
    train_loader, _ = build_dataloaders(config)
    inputs, _targets = next(iter(train_loader))

    for model_name in ["vgg16_bn", "resnet34", "mobilenetv2"]:
        config["model"]["name"] = model_name
        model = build_model(config)
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
        assert list(logits.shape) == [batch_size, num_classes]
