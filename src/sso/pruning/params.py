"""Parameter enumeration helpers for weight-level pruning."""

from __future__ import annotations

from collections.abc import Iterator

from torch import nn
from torch.nn import Parameter


def iter_prunable_named_parameters(
    model: nn.Module,
    include_bias: bool = False,
) -> Iterator[tuple[str, Parameter]]:
    """Yield prunable Conv2d and Linear parameters.

    By default this yields only ``weight`` parameters from ``nn.Conv2d`` and
    ``nn.Linear`` modules. Biases, normalization layers, embeddings, buffers,
    and classifier biases are excluded unless ``include_bias`` is explicitly
    enabled for Conv2d/Linear modules.
    """
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        prefix = f"{module_name}." if module_name else ""
        yield f"{prefix}weight", module.weight

        if include_bias and module.bias is not None:
            yield f"{prefix}bias", module.bias
