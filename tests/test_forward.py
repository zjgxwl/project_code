import torch

from sso.datasets import build_dataloaders
from sso.models import build_model


def test_cifar_resnet18_forward_shape_with_fake_data() -> None:
    batch_size = 4
    num_classes = 10
    config = {
        "dataset": {
            "name": "cifar10",
            "use_fake_data": True,
            "num_classes": num_classes,
            "image_size": 32,
            "batch_size": batch_size,
            "num_workers": 0,
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
