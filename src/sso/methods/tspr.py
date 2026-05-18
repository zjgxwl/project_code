"""Chapter 3 TSPR minimal training strategy."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import torch
from torch import nn

from sso.pruning import masked_state_dict
from .base import BaseSparseMethod


class TSPRMethod(BaseSparseMethod):
    """Training-signal-preserving sparse subnet parameter optimization.

    This minimal engineering version keeps the target mask fixed, does not
    hard-mask weights during training, and applies a regularizer only to the
    parameters that will be pruned in the exported sparse model.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_dict: Mapping[str, torch.Tensor],
        score_dict: Mapping[str, torch.Tensor],
        lambda0: float = 1e-4,
        eps: float = 1e-8,
        delta: float = 1e-12,
    ) -> None:
        super().__init__()
        self.model = model
        self.lambda0 = float(lambda0)
        self.eps = float(eps)
        self.delta = float(delta)

        named_parameters = dict(model.named_parameters())
        self.fixed_mask_dict: dict[str, torch.Tensor] = {}
        self.score_dict: dict[str, torch.Tensor] = {}
        self.initial_state: dict[str, torch.Tensor] = {}
        self.omega_dict: dict[str, torch.Tensor] = {}

        for name, mask in mask_dict.items():
            if name not in named_parameters:
                raise ValueError(f"Mask key does not match a model parameter: {name}")
            if name not in score_dict:
                raise ValueError(f"Missing score for masked parameter: {name}")

            parameter = named_parameters[name]
            if tuple(mask.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Mask shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            score = score_dict[name]
            if tuple(score.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Score shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(score.shape)}"
                )

            self.fixed_mask_dict[name] = mask.detach().clone().requires_grad_(False)
            self.score_dict[name] = score.detach().clone().requires_grad_(False)
            self.initial_state[name] = parameter.detach().clone().requires_grad_(False)
            self.omega_dict[name] = self._build_omega(score).requires_grad_(False)

    def _iter_masked_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        named_parameters = dict(self.model.named_parameters())
        for name in self.fixed_mask_dict:
            if name not in named_parameters:
                raise ValueError(f"Mask key does not match a model parameter: {name}")
            yield name, named_parameters[name]

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_omega(self, score: torch.Tensor) -> torch.Tensor:
        abs_score = score.detach().abs().clone()
        score_min = abs_score.min()
        score_max = abs_score.max()
        normalized = (abs_score - score_min) / (score_max - score_min + self.delta)
        omega = 1.0 / (normalized + self.eps)
        return torch.clamp(omega, max=1e6)

    def before_train(self) -> None:
        """TSPR does not hard-mask weights before training."""

    def lambda_at(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealed regularization strength."""
        if total_epochs <= 0:
            progress = 1.0
        else:
            progress = float(epoch) / float(total_epochs)
            progress = max(0.0, min(progress, 1.0))
        return self.lambda0 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def regularization_loss(self, epoch: int, total_epochs: int) -> torch.Tensor:
        """Return lambda-weighted TSPR regularization over pruned positions."""
        model_device = self._model_device()
        reg_loss = torch.zeros((), device=model_device)
        has_pruned_position = False

        for name, parameter in self._iter_masked_parameters():
            mask = self.fixed_mask_dict[name].to(device=parameter.device, dtype=parameter.dtype)
            pruned_mask = (mask == 0).to(dtype=parameter.dtype)
            if torch.count_nonzero(pruned_mask).item() == 0:
                continue

            has_pruned_position = True
            initial = self.initial_state[name].to(device=parameter.device, dtype=parameter.dtype)
            omega = self.omega_dict[name].to(device=parameter.device, dtype=parameter.dtype)
            diff = parameter - initial
            reg_loss = reg_loss + torch.sum(omega * pruned_mask * diff.square())

        if not has_pruned_position:
            return torch.zeros((), device=model_device)

        return reg_loss * self.lambda_at(epoch=epoch, total_epochs=total_epochs)

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Return masked model weights without mutating the training model."""
        return masked_state_dict(self.model, self.fixed_mask_dict)

    def method_state_dict(self) -> dict[str, object]:
        """Return TSPR method state, not ``model.state_dict()``."""
        return {
            "lambda0": self.lambda0,
            "eps": self.eps,
            "delta": self.delta,
            "fixed_mask_dict": {
                name: mask.detach().clone()
                for name, mask in self.fixed_mask_dict.items()
            },
            "omega_dict": {
                name: omega.detach().clone()
                for name, omega in self.omega_dict.items()
            },
        }

    def state_dict(self) -> dict[str, object]:
        """Return method state only; this is not ``model.state_dict()``."""
        return self.method_state_dict()
