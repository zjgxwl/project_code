from sso.datasets import build_dataloaders
from sso.metrics import accuracy_from_logits
from sso.methods import (
    BaseSparseMethod,
    EGROMethod,
    StandardSparseRetrainingMethod,
    TCSMMethod,
    TSPRMethod,
)
from sso.models import build_model, cifar_mobilenet_v2, cifar_resnet18, cifar_resnet34, cifar_vgg16_bn
from sso.training import evaluate, resolve_device, set_seed, train_one_epoch


def test_core_imports() -> None:
    assert build_dataloaders is not None
    assert build_model is not None
    assert cifar_resnet18 is not None
    assert cifar_resnet34 is not None
    assert cifar_mobilenet_v2 is not None
    assert cifar_vgg16_bn is not None
    assert train_one_epoch is not None
    assert evaluate is not None
    assert resolve_device is not None
    assert set_seed is not None
    assert accuracy_from_logits is not None


def test_sparse_method_placeholders_import() -> None:
    assert issubclass(TSPRMethod, BaseSparseMethod)
    assert issubclass(TCSMMethod, BaseSparseMethod)
    assert issubclass(EGROMethod, BaseSparseMethod)
    assert StandardSparseRetrainingMethod is not None


def test_base_sparse_method_lifecycle_defaults() -> None:
    method = BaseSparseMethod(config={"name": "demo"})

    assert method.is_setup is False
    assert method.setup() is method
    assert method.is_setup is True
    method.before_train()
    method.after_backward()
    method.after_optimizer_step()
    method.step()
    state = method.state_dict()
    assert state["config"] == {"name": "demo"}
    assert state["is_setup"] is True
    method.teardown()
    assert method.is_setup is False
