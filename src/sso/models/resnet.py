"""CIFAR-adapted ResNet models."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import resnet18

from .vgg import cifar_vgg11_bn


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 variant adapted for 32x32 CIFAR inputs."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from a project configuration dictionary."""
    model_config = config.get("model", {})
    name = model_config.get("name", "resnet18").lower()
    num_classes = int(model_config.get("num_classes", config.get("dataset", {}).get("num_classes", 10)))

    if name == "resnet18":
        return cifar_resnet18(num_classes=num_classes)
    if name == "vgg11_bn":
        return cifar_vgg11_bn(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
