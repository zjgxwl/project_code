"""Small CIFAR VGG-style models for structured export smoke tests."""

from __future__ import annotations

import torch
from torch import nn


class CifarVGG(nn.Module):
    """A project-local serial Conv-BN-ReLU VGG-style network."""

    def __init__(
        self,
        channels: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("CifarVGG requires at least one convolution channel.")

        layers: list[nn.Module] = []
        current_in_channels = int(in_channels)
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        current_in_channels,
                        int(out_channels),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(int(out_channels)),
                    nn.ReLU(inplace=True),
                ]
            )
            current_in_channels = int(out_channels)

        self.channels = [int(channel) for channel in channels]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.channels[-1], int(num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def cifar_vgg11_bn(num_classes: int = 10) -> CifarVGG:
    """Build the lightweight project-local CIFAR VGG-11-BN variant."""
    return CifarVGG(
        channels=[32, 32, 64, 64, 128, 128, 256, 256],
        num_classes=num_classes,
    )
