"""Approximate Hessian metrics for small analysis runs."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _first_loss(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    total_loss: torch.Tensor | None = None
    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = criterion(model(inputs), targets)
        total_loss = loss if total_loss is None else total_loss + loss
        batch_count += 1
    if total_loss is None or batch_count == 0:
        raise ValueError("dataloader yielded no batches for Hessian computation.")
    return total_loss / batch_count


def _hessian_vector_product(
    loss: torch.Tensor,
    parameters: list[nn.Parameter],
    vectors: list[torch.Tensor],
) -> list[torch.Tensor]:
    grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    grad_vector_product = torch.zeros((), device=loss.device)
    for grad, vector in zip(grads, vectors, strict=True):
        if grad is not None:
            grad_vector_product = grad_vector_product + torch.sum(grad * vector)
    hvp = torch.autograd.grad(grad_vector_product, parameters, retain_graph=True, allow_unused=True)
    return [
        torch.zeros_like(parameter) if value is None else value
        for parameter, value in zip(parameters, hvp, strict=True)
    ]


def hutchinson_trace(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device | str,
    num_samples: int = 1,
    max_batches: int = 1,
) -> float:
    """Estimate Hessian trace with Rademacher Hutchinson probes."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    resolved_device = torch.device(device)
    was_training = model.training
    try:
        model.eval()
        parameters = _trainable_parameters(model)
        loss = _first_loss(model, dataloader, criterion, resolved_device, max_batches)
        estimates: list[float] = []
        for _sample_idx in range(num_samples):
            vectors = [
                torch.randint_like(parameter, low=0, high=2, dtype=parameter.dtype) * 2 - 1
                for parameter in parameters
            ]
            hvp = _hessian_vector_product(loss, parameters, vectors)
            estimate = sum(torch.sum(vector * product) for vector, product in zip(vectors, hvp, strict=True))
            estimates.append(float(estimate.detach().cpu().item()))
        return sum(estimates) / len(estimates)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def largest_hessian_eigenvalue(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device | str,
    num_iters: int = 10,
    max_batches: int = 1,
) -> float:
    """Estimate the largest Hessian eigenvalue with power iteration."""
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")
    resolved_device = torch.device(device)
    was_training = model.training
    try:
        model.eval()
        parameters = _trainable_parameters(model)
        loss = _first_loss(model, dataloader, criterion, resolved_device, max_batches)
        vectors = [torch.randn_like(parameter) for parameter in parameters]
        eigenvalue = 0.0
        for _iter_idx in range(num_iters):
            norm = torch.sqrt(sum(torch.sum(vector.square()) for vector in vectors))
            if float(norm.detach().cpu().item()) == 0.0:
                return 0.0
            vectors = [vector / norm for vector in vectors]
            hvp = _hessian_vector_product(loss, parameters, vectors)
            eigenvalue_tensor = sum(torch.sum(vector * product) for vector, product in zip(vectors, hvp, strict=True))
            eigenvalue = float(eigenvalue_tensor.detach().cpu().item())
            vectors = [product.detach() for product in hvp]
        return eigenvalue
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
