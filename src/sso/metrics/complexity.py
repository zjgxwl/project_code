"""Model-size and FLOPs helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(int(parameter.numel()) for parameter in parameters)


def estimate_conv2d_flops(
    model: nn.Module,
    input_shape: Sequence[int],
    device: torch.device | str | None = None,
) -> float:
    """Estimate Conv2d multiply-add FLOPs with a single dummy forward pass."""
    resolved_device = torch.device(device) if device is not None else next(model.parameters()).device
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    total_flops = 0.0
    was_training = model.training

    def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total_flops
        if not isinstance(module, nn.Conv2d):
            return
        if output.ndim != 4:
            raise ValueError("Conv2d output must be 4D for FLOPs estimation.")
        batch, out_channels, h_out, w_out = output.shape
        k_h, k_w = module.kernel_size
        groups = int(module.groups)
        per_output = (int(module.in_channels) // groups) * int(k_h) * int(k_w)
        total_flops += float(batch * out_channels * h_out * w_out * per_output * 2)

    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook))
        model.eval()
        dummy = torch.ones(tuple(int(dim) for dim in input_shape), device=resolved_device)
        with torch.no_grad():
            model(dummy)
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)

    return total_flops
