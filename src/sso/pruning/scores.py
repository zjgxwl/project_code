"""Pruning score functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .params import iter_prunable_named_parameters


def magnitude_score(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return absolute weight magnitude scores for prunable parameters."""
    return {
        name: parameter.detach().abs().clone()
        for name, parameter in iter_prunable_named_parameters(model)
    }


def _restore_training_mode(model: nn.Module, was_training: bool) -> None:
    if was_training:
        model.train()
    else:
        model.eval()


def _prunable_grad_scores(model: nn.Module) -> dict[str, torch.Tensor]:
    score_dict: dict[str, torch.Tensor] = {}
    for name, parameter in iter_prunable_named_parameters(model):
        if parameter.grad is None:
            score = torch.zeros_like(parameter)
        else:
            score = (parameter * parameter.grad).detach().abs().clone()
        score_dict[name] = score
    return score_dict


def snip_score(
    model: nn.Module,
    dataloader: Iterable | None,
    criterion: nn.Module | None,
    device: torch.device | str | None,
    max_batches: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute SNIP scores as ``abs(weight * grad)`` over a few batches."""
    if dataloader is None:
        raise ValueError("snip_score requires a dataloader.")
    if criterion is None:
        raise ValueError("snip_score requires a criterion.")
    if device is None:
        raise ValueError("snip_score requires a device.")

    device = torch.device(device)
    was_training = model.training
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    model.zero_grad(set_to_none=True)
    model.train()

    try:
        total_loss: torch.Tensor | None = None
        batch_count = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss = loss if total_loss is None else total_loss + loss
            batch_count += 1

        if total_loss is None or batch_count == 0:
            raise ValueError("snip_score received no batches from dataloader.")

        average_loss = total_loss / batch_count
        average_loss.backward()
        return _prunable_grad_scores(model)
    finally:
        with torch.no_grad():
            model.load_state_dict(original_state)
        model.zero_grad(set_to_none=True)
        _restore_training_mode(model, was_training)


def synflow_score(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | str | None,
) -> dict[str, torch.Tensor]:
    """Compute SynFlow scores using all-one inputs and linearized weights."""
    if device is None:
        raise ValueError("synflow_score requires a device.")
    if input_shape is None:
        raise ValueError("synflow_score requires an input_shape.")

    device = torch.device(device)
    was_training = model.training
    original_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    model.zero_grad(set_to_none=True)
    model.eval()

    try:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.abs_()

        inputs = torch.ones(tuple(input_shape), device=device)
        output = model(inputs).sum()
        output.backward()
        return _prunable_grad_scores(model)
    finally:
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), original_parameters, strict=True):
                parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))
        model.zero_grad(set_to_none=True)
        _restore_training_mode(model, was_training)


def grasp_score(
    model: nn.Module,
    dataloader: Iterable | None,
    criterion: nn.Module | None,
    device: torch.device | str | None,
    max_batches: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute a minimal GraSP-style second-order score."""
    if dataloader is None:
        raise ValueError("grasp_score requires a dataloader.")
    if criterion is None:
        raise ValueError("grasp_score requires a criterion.")
    if device is None:
        raise ValueError("grasp_score requires a device.")

    device = torch.device(device)
    was_training = model.training
    saved_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    model.zero_grad(set_to_none=True)
    model.train()

    try:
        prunable_items = list(iter_prunable_named_parameters(model))
        prunable_parameters = [parameter for _name, parameter in prunable_items]

        total_loss: torch.Tensor | None = None
        batch_count = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss = loss if total_loss is None else total_loss + loss
            batch_count += 1

        if total_loss is None or batch_count == 0:
            raise ValueError("grasp_score received no batches from dataloader.")

        average_loss = total_loss / batch_count
        first_grads = torch.autograd.grad(
            average_loss,
            prunable_parameters,
            create_graph=True,
            allow_unused=True,
        )
        non_none_grads = [grad for grad in first_grads if grad is not None]
        if not non_none_grads:
            return {
                name: torch.zeros_like(parameter)
                for name, parameter in prunable_items
            }

        grad_flow = sum(grad.pow(2).sum() for grad in non_none_grads)
        second_grads = torch.autograd.grad(
            grad_flow,
            prunable_parameters,
            allow_unused=True,
        )

        score_dict: dict[str, torch.Tensor] = {}
        for (name, parameter), first_grad, second_grad in zip(
            prunable_items,
            first_grads,
            second_grads,
            strict=True,
        ):
            if first_grad is None or second_grad is None:
                score = torch.zeros_like(parameter)
            else:
                score = (parameter * second_grad).detach().abs().clone()
            score_dict[name] = score
        return score_dict
    finally:
        model.load_state_dict(saved_state, strict=True)
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def ep_score(*_args: Any, **_kwargs: Any) -> dict[str, torch.Tensor]:
    """Placeholder for future EP scores."""
    raise NotImplementedError("TODO: implement EP scoring in a future chapter method.")
