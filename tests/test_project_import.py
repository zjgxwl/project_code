from sso.datasets import build_dataloaders
from sso.methods import (
    BaseSparseMethod,
    EGROMethod,
    StandardSparseRetrainingMethod,
    TCSMMethod,
    TSPRMethod,
)
from sso.models import build_model, cifar_resnet18
from sso.training import evaluate, resolve_device, set_seed, train_one_epoch


def test_core_imports() -> None:
    assert build_dataloaders is not None
    assert build_model is not None
    assert cifar_resnet18 is not None
    assert train_one_epoch is not None
    assert evaluate is not None
    assert resolve_device is not None
    assert set_seed is not None


def test_sparse_method_placeholders_import() -> None:
    assert issubclass(TSPRMethod, BaseSparseMethod)
    assert issubclass(TCSMMethod, BaseSparseMethod)
    assert issubclass(EGROMethod, BaseSparseMethod)
    assert StandardSparseRetrainingMethod is not None
