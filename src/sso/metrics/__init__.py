"""Metric helpers shared by thesis experiment runners."""

from .classification import accuracy_from_logits
from .complexity import count_parameters, estimate_conv2d_flops
from .hessian import hutchinson_trace, largest_hessian_eigenvalue
from .loss_analysis import linear_interpolation_losses, loss_barrier
from .ranking import score_rank_correlation
from .sparsity import mask_dict_sparsity, mask_jaccard, tensor_sparsity
from .stats import confidence_interval_95, mean, sample_std, welch_t_statistic

__all__ = [
    "accuracy_from_logits",
    "confidence_interval_95",
    "count_parameters",
    "estimate_conv2d_flops",
    "hutchinson_trace",
    "largest_hessian_eigenvalue",
    "linear_interpolation_losses",
    "loss_barrier",
    "mask_dict_sparsity",
    "mask_jaccard",
    "mean",
    "sample_std",
    "score_rank_correlation",
    "tensor_sparsity",
    "welch_t_statistic",
]

__all__: list[str] = []
