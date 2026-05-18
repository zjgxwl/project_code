"""Mask creation and application utilities for weight-level pruning."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def _validate_sparsity(sparsity: float) -> None:
    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")


def global_topk_mask(
    score_dict: Mapping[str, torch.Tensor],
    sparsity: float,
) -> dict[str, torch.Tensor]:
    """Create global Top-K masks from pruning scores.

    ``sparsity=0.9`` means pruning 90% of scored parameters and keeping 10%.
    """
    _validate_sparsity(sparsity)

    total_params = sum(score.numel() for score in score_dict.values())
    if total_params == 0:
        return {}

    num_keep = int(round((1.0 - sparsity) * total_params))
    num_keep = max(0, min(num_keep, total_params))

    if num_keep == total_params:
        return {name: torch.ones_like(score) for name, score in score_dict.items()}
    if num_keep == 0:
        return {name: torch.zeros_like(score) for name, score in score_dict.items()}

    flat_scores = torch.cat(
        [score.detach().reshape(-1).float().cpu() for score in score_dict.values()]
    )
    topk_indices = torch.topk(flat_scores, k=num_keep, largest=True, sorted=False).indices
    flat_mask = torch.zeros(total_params, dtype=torch.bool)
    flat_mask[topk_indices] = True

    mask_dict: dict[str, torch.Tensor] = {}
    offset = 0
    for name, score in score_dict.items():
        next_offset = offset + score.numel()
        mask = flat_mask[offset:next_offset].reshape(score.shape)
        mask_dict[name] = mask.to(device=score.device, dtype=score.dtype)
        offset = next_offset

    return mask_dict


def compute_sparsity(mask_dict: Mapping[str, torch.Tensor]) -> float:
    """Compute sparsity over the tensors present in ``mask_dict`` only."""
    total_params = sum(mask.numel() for mask in mask_dict.values())
    if total_params == 0:
        return 0.0

    kept_params = sum(int(torch.count_nonzero(mask).item()) for mask in mask_dict.values())
    return 1.0 - kept_params / total_params


def apply_mask_to_model(
    model: nn.Module,
    mask_dict: Mapping[str, torch.Tensor],
) -> nn.Module:
    """Apply masks to model parameters in place."""
    named_parameters = dict(model.named_parameters())

    with torch.no_grad():
        for name, mask in mask_dict.items():
            if name not in named_parameters:
                raise ValueError(f"Mask key does not match a model parameter: {name}")

            parameter = named_parameters[name]
            if tuple(mask.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Mask shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(mask.shape)}"
                )

            parameter.mul_(mask.to(device=parameter.device, dtype=parameter.dtype))

    return model


def masked_state_dict(
    model: nn.Module,
    mask_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a full cloned state dict with masks applied to selected parameters."""
    named_parameters = dict(model.named_parameters())
    cloned_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    for name, mask in mask_dict.items():
        if name not in named_parameters:
            raise ValueError(f"Mask key does not match a model parameter: {name}")

        parameter = named_parameters[name]
        if tuple(mask.shape) != tuple(parameter.shape):
            raise ValueError(
                f"Mask shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                f"got {tuple(mask.shape)}"
            )

        cloned_state[name] = parameter.detach().clone() * mask.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

    return cloned_state
