"""Sparsity and mask-overlap metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def tensor_sparsity(tensor: torch.Tensor) -> float:
    """Return the fraction of zero entries in a tensor."""
    total = int(tensor.numel())
    if total == 0:
        return 0.0
    zeros = int(torch.count_nonzero(tensor.detach() == 0).item())
    return zeros / total


def mask_dict_sparsity(mask_dict: Mapping[str, torch.Tensor]) -> float:
    """Return global sparsity for a dictionary of binary masks."""
    total = 0
    zeros = 0
    for mask in mask_dict.values():
        detached = mask.detach()
        total += int(detached.numel())
        zeros += int(torch.count_nonzero(detached == 0).item())
    if total == 0:
        return 0.0
    return zeros / total


def mask_jaccard(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> float:
    """Return Jaccard similarity between two binary mask dictionaries."""
    if set(first) != set(second):
        missing_first = sorted(set(second) - set(first))
        missing_second = sorted(set(first) - set(second))
        raise ValueError(
            "Mask dictionaries must have identical keys. "
            f"Missing from first: {missing_first}; missing from second: {missing_second}."
        )

    intersection = 0
    union = 0
    for name, first_mask in first.items():
        second_mask = second[name]
        if tuple(first_mask.shape) != tuple(second_mask.shape):
            raise ValueError(f"Mask shape mismatch for {name}.")
        first_bool = first_mask.detach().to(dtype=torch.bool).flatten().cpu()
        second_bool = second_mask.detach().to(dtype=torch.bool).flatten().cpu()
        intersection += int(torch.logical_and(first_bool, second_bool).sum().item())
        union += int(torch.logical_or(first_bool, second_bool).sum().item())

    if union == 0:
        return 1.0
    return intersection / union
