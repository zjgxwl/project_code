"""Factory for pruning score dictionaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from .scores import ep_score, grasp_score, magnitude_score, snip_score, synflow_score


def _resolve_scorer(config: Mapping[str, Any], scorer: str | None = None) -> str:
    if scorer is not None:
        return scorer.lower()
    return str(config.get("pruning", {}).get("scorer", "magnitude")).lower()


def _infer_input_shape(
    config: Mapping[str, Any],
    dataloader: Iterable | None,
) -> tuple[int, ...]:
    pruning_config = config.get("pruning", {})
    configured_shape = pruning_config.get("input_shape")
    if configured_shape is not None:
        return tuple(int(dim) for dim in configured_shape)

    if dataloader is not None:
        try:
            inputs, _targets = next(iter(dataloader))
            return tuple(int(dim) for dim in inputs.shape)
        except StopIteration:
            pass

    dataset_config = config.get("dataset", {})
    image_size = int(dataset_config.get("image_size", 32))
    return (1, 3, image_size, image_size)


def build_score_dict(
    model: nn.Module,
    config: Mapping[str, Any],
    dataloader: Iterable | None = None,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    scorer: str | None = None,
) -> dict[str, torch.Tensor]:
    """Build a pruning score dictionary from config or explicit scorer."""
    scorer_name = _resolve_scorer(config, scorer=scorer)
    pruning_config = config.get("pruning", {})

    if scorer_name == "magnitude":
        return magnitude_score(model)
    if scorer_name == "snip":
        max_batches = int(pruning_config.get("score_batches", 1))
        return snip_score(
            model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )
    if scorer_name == "synflow":
        input_shape = _infer_input_shape(config, dataloader)
        return synflow_score(model, input_shape=input_shape, device=device)
    if scorer_name == "grasp":
        return grasp_score(model)
    if scorer_name == "ep":
        return ep_score(model)
    raise ValueError(f"Unsupported pruning scorer: {scorer_name}")
