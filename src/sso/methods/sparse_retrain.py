"""Standard fixed-mask sparse retraining baseline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
from torch import nn

from sso.pruning import apply_mask_to_model, masked_state_dict


class StandardSparseRetrainingMethod:
    """Fixed-mask sparse retraining baseline.

    The pruning masks are fixed method state. They are not model parameters and
    do not participate in gradient updates.
    """

    def __init__(self, model: nn.Module, mask_dict: Mapping[str, torch.Tensor]) -> None:
        self.model = model
        self.mask_dict = {
            name: mask.detach().clone().requires_grad_(False)
            for name, mask in mask_dict.items()
        }

    def _iter_masked_parameters(self) -> Iterator[tuple[str, nn.Parameter, torch.Tensor]]:
        named_parameters = dict(self.model.named_parameters())
        for name, mask in self.mask_dict.items():
            if name not in named_parameters:
                raise ValueError(f"Mask key does not match a model parameter: {name}")

            parameter = named_parameters[name]
            if tuple(mask.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Mask shape mismatch for {name}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            yield name, parameter, mask

    def apply_mask(self) -> nn.Module:
        """Apply the fixed masks to model parameters in place."""
        return apply_mask_to_model(self.model, self.mask_dict)

    def before_train(self) -> nn.Module:
        """Apply the fixed masks before training starts."""
        return self.apply_mask()

    def mask_gradients(self) -> None:
        """Mask gradients after backward and before optimizer step."""
        with torch.no_grad():
            for _name, parameter, mask in self._iter_masked_parameters():
                if parameter.grad is None:
                    continue
                parameter.grad.mul_(mask.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def after_backward(self) -> None:
        """Hook alias used by the training loop after loss.backward()."""
        self.mask_gradients()

    def after_optimizer_step(self) -> nn.Module:
        """Re-apply masks after optimizer updates."""
        return self.apply_mask()

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a full masked model state dict without mutating the model."""
        return masked_state_dict(self.model, self.mask_dict)

    def mask_state_dict(self) -> dict[str, torch.Tensor]:
        """Return cloned fixed masks owned by this method."""
        return {name: mask.detach().clone() for name, mask in self.mask_dict.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return method state only: fixed masks, not ``model.state_dict()``."""
        return self.mask_state_dict()
