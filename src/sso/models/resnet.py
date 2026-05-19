"""CIFAR-adapted ResNet models."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import mobilenet_v2, resnet18, resnet34

from .vgg import cifar_vgg11_bn, cifar_vgg16_bn


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 variant adapted for 32x32 CIFAR inputs."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def cifar_resnet34(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-34 variant adapted for 32x32 CIFAR inputs."""
    model = resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def cifar_mobilenet_v2(num_classes: int = 10) -> nn.Module:
    """Build a MobileNetV2 variant adapted for 32x32 CIFAR inputs."""
    model = mobilenet_v2(weights=None)
    first = model.features[0][0]
    if isinstance(first, nn.Conv2d):
        model.features[0][0] = nn.Conv2d(
            3,
            first.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from a project configuration dictionary."""
    model_config = config.get("model", {})
    name = model_config.get("name", "resnet18").lower()
    num_classes = int(model_config.get("num_classes", config.get("dataset", {}).get("num_classes", 10)))

    if name == "resnet18":
        return cifar_resnet18(num_classes=num_classes)
    if name == "resnet34":
        return cifar_resnet34(num_classes=num_classes)
    if name in {"mobilenetv2", "mobilenet_v2"}:
        return cifar_mobilenet_v2(num_classes=num_classes)
    if name == "vgg11_bn":
        return cifar_vgg11_bn(num_classes=num_classes)
    if name == "vgg16_bn":
        return cifar_vgg16_bn(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
