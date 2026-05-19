"""Unified thesis experiment entry point with structured logging."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sso.datasets import build_calibration_dataloader, build_dataloaders
from sso.export import export_vgg_slim_artifact
from sso.metrics import count_parameters, estimate_conv2d_flops
from sso.methods import EGROTrainingMethod, StandardSparseRetrainingMethod, TSPRMethod
from sso.models import build_model
from sso.pruning import build_score_dict, compute_sparsity, global_topk_mask
from sso.training import evaluate, fit_model, resolve_device, set_seed
from sso.utils import RunLogger, append_jsonl, build_run_name, dataset_record_fields, flatten_metrics


DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "cifar10": {"num_classes": 10, "image_size": 32},
    "cifar100": {"num_classes": 100, "image_size": 32},
    "tiny_imagenet": {"num_classes": 200, "image_size": 64},
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return loaded


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for record in records for field in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config.get("optimizer", {})
    name = str(optimizer_config.get("name", "sgd")).lower()
    lr = float(optimizer_config.get("lr", 0.01))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    if name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any | None:
    scheduler_config = config.get("scheduler", {})
    name = str(scheduler_config.get("name", "cosine")).lower()
    if name in {"none", "disabled"}:
        return None
    if name == "cosine":
        epochs = max(1, int(config.get("training", {}).get("epochs", 1)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {name}")


def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = config.get("profiles", {})
    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(profiles)}")
    return selected


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _status_for(config: dict[str, Any], kind: str, name: str) -> str:
    return str(config.get("capabilities", {}).get(kind, {}).get(name, "supported"))


def _normalize_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    return "tiny_imagenet" if normalized == "tinyimagenet" else normalized


def build_experiments(config: dict[str, Any], profile: str, chapter: str) -> list[dict[str, Any]]:
    selected = profile_config(config, profile)
    dataset_entries = _as_list(selected.get("datasets", selected.get("dataset", {})))
    base = {
        "chapter": chapter,
        "profile": profile,
        "optimizer": selected.get("optimizer", {}),
        "scheduler": selected.get("scheduler", {"name": "cosine"}),
        "training": selected.get("training", {}),
        "tcsm": selected.get("tcsm", {}),
        "egro": selected.get("egro", {}),
        "device": selected.get("device", "auto"),
    }
    experiments: list[dict[str, Any]] = []
    for dataset, seed, model, scorer in product(
        dataset_entries,
        _as_list(selected.get("seeds", [42])),
        _as_list(selected.get("models", ["resnet18"])),
        _as_list(selected.get("scorers", ["magnitude"])),
    ):
        for method in _as_list(selected.get("methods", [chapter])):
            sparsities = _as_list(selected.get("sparsities", [None]))
            flops_reductions = _as_list(selected.get("flops_reductions", [None]))
            for sparsity, flops_reduction in product(sparsities, flops_reductions):
                experiment = {
                    **base,
                    "dataset": dataset,
                    "seed": int(seed),
                    "model": str(model),
                    "method": str(method),
                    "scorer": str(scorer),
                    "sparsity": None if sparsity is None else float(sparsity),
                    "flops_reduction": None if flops_reduction is None else float(flops_reduction),
                    "status": "supported",
                }
                model_status = _status_for(config, "models", experiment["model"])
                method_status = _status_for(config, "methods", experiment["method"])
                if model_status != "supported":
                    experiment["status"] = model_status
                if method_status != "supported":
                    experiment["status"] = method_status
                experiments.append(experiment)
    return experiments


def apply_cli_overrides(
    experiments: list[dict[str, Any]],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Apply traditional training CLI overrides to experiment records."""
    overridden: list[dict[str, Any]] = []
    for experiment in experiments:
        updated = dict(experiment)
        if args.seed is not None:
            updated["seed"] = int(args.seed)
        if args.model is not None:
            updated["model"] = args.model.lower()
        if args.method is not None:
            updated["method"] = args.method.lower()
        if args.scorer is not None:
            updated["scorer"] = args.scorer.lower()
        if args.sparsity is not None:
            updated["sparsity"] = float(args.sparsity)
        if args.flops_reduction is not None:
            updated["flops_reduction"] = float(args.flops_reduction)

        dataset_config = dict(updated["dataset"])
        if args.dataset is not None:
            dataset_name = _normalize_dataset_name(args.dataset)
            dataset_config["name"] = dataset_name
            dataset_config.update(DATASET_DEFAULTS.get(dataset_name, {}))
        if args.batch_size is not None:
            dataset_config["batch_size"] = int(args.batch_size)
        updated["dataset"] = dataset_config

        optimizer_config = dict(updated["optimizer"])
        if args.lr is not None:
            optimizer_config["lr"] = float(args.lr)
        updated["optimizer"] = optimizer_config

        training_config = dict(updated["training"])
        if args.epochs is not None:
            training_config["epochs"] = int(args.epochs)
        if args.max_train_batches is not None:
            training_config["max_train_batches"] = int(args.max_train_batches)
        if args.max_val_batches is not None:
            training_config["max_val_batches"] = int(args.max_val_batches)
        if args.log_batches:
            training_config["log_batches"] = True
        if args.log_interval is not None:
            training_config["log_interval"] = int(args.log_interval)
        updated["training"] = training_config

        updated["status"] = "supported"
        model_status = _status_for(config, "models", updated["model"])
        method_status = _status_for(config, "methods", updated["method"])
        if model_status != "supported":
            updated["status"] = model_status
        if method_status != "supported":
            updated["status"] = method_status
        overridden.append(updated)
    return overridden


def experiment_to_config(experiment: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = dict(experiment["dataset"])
    if args.data_dir is not None:
        dataset_config["data_dir"] = str(args.data_dir)
    if args.download:
        dataset_config["download"] = True
    config = {
        "device": experiment["device"],
        "seed": experiment["seed"],
        "dataset": dataset_config,
        "model": {
            "name": experiment["model"],
            "num_classes": dataset_config.get("num_classes", 10),
        },
        "optimizer": dict(experiment["optimizer"]),
        "scheduler": dict(experiment["scheduler"]),
        "training": dict(experiment["training"]),
        "pruning": {
            "scorer": experiment["scorer"],
            "sparsity": 0.9 if experiment["sparsity"] is None else experiment["sparsity"],
            "score_batches": int(experiment.get("score_batches", 1)),
        },
        "method": dict(experiment.get("method_config", {})),
        "tcsm": dict(experiment["tcsm"]),
        "egro": dict(experiment["egro"]),
    }
    config["method"].setdefault("lambda0", 1e-4)
    config["method"].setdefault("eps", 1e-8)
    config["method"].setdefault("delta", 1e-12)
    if experiment["chapter"] == "chapter4" and experiment["method"] in {"base_only", "random_subset", "full_data"}:
        config["tcsm"]["calibration_mode"] = experiment["method"]
    if experiment["flops_reduction"] is not None:
        config["egro"]["flops_reduction"] = experiment["flops_reduction"]
    return config


def bounded_batches(config: dict[str, Any]) -> tuple[int | None, int | None]:
    training = config.get("training", {})
    max_train_batches = training.get("max_train_batches")
    max_val_batches = training.get("max_val_batches")
    return (
        None if max_train_batches is None else int(max_train_batches),
        None if max_val_batches is None else int(max_val_batches),
    )


def model_complexity_metrics(model: nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    image_size = int(config.get("dataset", {}).get("image_size", 32))
    input_shape = (1, 3, image_size, image_size)
    try:
        conv_flops = estimate_conv2d_flops(model, input_shape=input_shape)
    except Exception:
        conv_flops = None
    return {
        "model_params": count_parameters(model),
        "model_trainable_params": count_parameters(model, trainable_only=True),
        "model_conv_flops": conv_flops,
    }


def mask_summary(mask_dict: dict[str, torch.Tensor], target_sparsity: float | None) -> dict[str, Any]:
    total = sum(int(mask.numel()) for mask in mask_dict.values())
    kept = sum(int(mask.count_nonzero().item()) for mask in mask_dict.values())
    pruned = total - kept
    actual = 0.0 if total == 0 else pruned / total
    return {
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual,
        "kept_params": kept,
        "pruned_params": pruned,
        "mask_total_params": total,
        "param_reduction": actual,
    }


def base_run_metrics(experiment: dict[str, Any], config: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    return {
        "chapter": experiment["chapter"],
        "profile": experiment["profile"],
        "method": experiment["method"],
        "dataset": config.get("dataset", {}).get("name"),
        "model": experiment["model"],
        "seed": int(config["seed"]),
        "scorer": experiment["scorer"],
        "batch_size": int(config.get("dataset", {}).get("batch_size", 0)),
        "target_sparsity": experiment["sparsity"],
        "target_flops_reduction": experiment["flops_reduction"],
        **model_complexity_metrics(model, config),
    }


def create_logger(
    args: argparse.Namespace,
    experiment: dict[str, Any],
    config: dict[str, Any],
    run_name: str,
) -> RunLogger:
    chapter_dir = args.output_dir / experiment["chapter"]
    return RunLogger(
        chapter_dir,
        run_name=run_name,
        config=config,
        command=sys.argv,
        metadata={
            "chapter": experiment["chapter"],
            "profile": experiment["profile"],
            "method": experiment["method"],
        },
    )


def final_sparse_eval(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    val_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    max_val_batches: int | None,
) -> dict[str, float]:
    eval_model = build_model(config).to(device)
    eval_model.load_state_dict(state_dict)
    return evaluate(eval_model, val_loader, criterion, device, max_batches=max_val_batches)


def run_chapter3(args: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment_to_config(experiment, args)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    score_dict = build_score_dict(model, config, dataloader=train_loader, criterion=criterion, device=device)
    mask_dict = global_topk_mask(score_dict, sparsity=float(config["pruning"]["sparsity"]))
    run_metrics = {
        **base_run_metrics(experiment, config, model),
        **mask_summary(mask_dict, experiment["sparsity"]),
    }
    method_name = experiment["method"]
    if method_name == "standard":
        sparse_method: Any = StandardSparseRetrainingMethod(model, mask_dict)
    elif method_name == "tspr":
        sparse_method = TSPRMethod(
            model,
            mask_dict,
            score_dict,
            lambda0=float(config["method"]["lambda0"]),
            eps=float(config["method"]["eps"]),
            delta=float(config["method"]["delta"]),
        )
    else:
        raise ValueError(f"Unsupported chapter3 method for execution: {method_name}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    max_train_batches, max_val_batches = bounded_batches(config)
    run_name = build_run_name(method=f"chapter3_{method_name}", scorer=experiment["scorer"], model=experiment["model"], sparsity=experiment["sparsity"])
    logger = create_logger(args, experiment, config, run_name)
    fit_result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(config["training"].get("epochs", 1)),
        method=sparse_method,
        scheduler=scheduler,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        eval_interval=int(config["training"].get("eval_interval", 1)),
        checkpoint_interval=int(config["training"].get("checkpoint_interval", 0)),
        logger=logger,
        base_metrics=run_metrics,
        log_batches=bool(config["training"].get("log_batches", False)),
        log_interval=int(config["training"].get("log_interval", 1)),
    )
    export_state = sparse_method.export_state_dict()
    export_metrics = final_sparse_eval(config, export_state, val_loader, criterion, device, max_val_batches)
    summary = {
        **experiment,
        **dataset_record_fields(config),
        "run_dir": logger.run_dir,
        **run_metrics,
        "sparsity_actual": compute_sparsity(mask_dict),
        "epochs_ran": fit_result.epochs,
        "train_loss": fit_result.final_train["loss"],
        "train_acc": fit_result.final_train["accuracy"],
        "val_loss": export_metrics["loss"],
        "val_acc": export_metrics["accuracy"],
        "best_val_acc": fit_result.best_val_accuracy,
    }
    logger.write_summary(summary)
    return summary


def build_tcsm_output(config: dict[str, Any], model: nn.Module, train_loader: Any, criterion: nn.Module, device: torch.device) -> Any:
    from sso.methods import TCSMMethod

    base_score_dict = build_score_dict(model, config, dataloader=train_loader, criterion=criterion, device=device)
    calibration_loader = build_calibration_dataloader(config, train_loader)
    tcsm_config = config.get("tcsm", {})
    tcsm = TCSMMethod(
        model=model,
        base_score_dict=base_score_dict,
        dataloader=calibration_loader,
        criterion=criterion,
        device=device,
        sparsity=float(config["pruning"]["sparsity"]),
        alpha=float(tcsm_config.get("alpha", 0.5)),
        beta=float(tcsm_config.get("beta", 0.5)),
        eta=float(tcsm_config.get("eta", 0.05)),
        delta=float(tcsm_config.get("delta", 1e-12)),
        calibration_batches=int(tcsm_config.get("calibration_batches", 1)),
        sign_batches=int(tcsm_config.get("sign_batches", 1)),
        calibration_mode=str(tcsm_config.get("calibration_mode", "tcsm")),
        stability_repeats=int(tcsm_config.get("stability_repeats", 1)),
    )
    return tcsm.run()


def run_chapter4(args: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment_to_config(experiment, args)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    tcsm_output = build_tcsm_output(config, model, train_loader, criterion, device)
    run_metrics = {
        **base_run_metrics(experiment, config, model),
        **mask_summary(tcsm_output.stable_mask_dict, experiment["sparsity"]),
        **tcsm_output.metrics,
    }
    sparse_method = TSPRMethod(
        model,
        tcsm_output.stable_mask_dict,
        tcsm_output.stable_score_dict,
        lambda0=float(config["method"]["lambda0"]),
        eps=float(config["method"]["eps"]),
        delta=float(config["method"]["delta"]),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    max_train_batches, max_val_batches = bounded_batches(config)
    run_name = build_run_name(method="chapter4_tcsm_tspr", scorer=experiment["scorer"], model=experiment["model"], sparsity=experiment["sparsity"])
    logger = create_logger(args, experiment, config, run_name)
    fit_result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(config["training"].get("epochs", 1)),
        method=sparse_method,
        scheduler=scheduler,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        eval_interval=int(config["training"].get("eval_interval", 1)),
        checkpoint_interval=int(config["training"].get("checkpoint_interval", 0)),
        logger=logger,
        base_metrics=run_metrics,
        log_batches=bool(config["training"].get("log_batches", False)),
        log_interval=int(config["training"].get("log_interval", 1)),
        extra_epoch_metrics=lambda _epoch, _total: tcsm_output.metrics,
    )
    export_metrics = final_sparse_eval(config, sparse_method.export_state_dict(), val_loader, criterion, device, max_val_batches)
    summary = {
        **experiment,
        **dataset_record_fields(config),
        "run_dir": logger.run_dir,
        **run_metrics,
        "epochs_ran": fit_result.epochs,
        "train_loss": fit_result.final_train["loss"],
        "train_acc": fit_result.final_train["accuracy"],
        "val_loss": export_metrics["loss"],
        "val_acc": export_metrics["accuracy"],
    }
    logger.write_summary(summary)
    return summary


def run_chapter5(args: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    from sso.methods import EGROMethod

    config = experiment_to_config(experiment, args)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    tcsm_output = build_tcsm_output(config, model, train_loader, criterion, device)
    base_metrics = base_run_metrics(experiment, config, model)
    egro_config = config.get("egro", {})
    image_size = int(config.get("dataset", {}).get("image_size", 32))
    input_shape = egro_config.get("input_shape", [1, 3, image_size, image_size])
    egro = EGROMethod(
        model=model,
        stable_score_dict=tcsm_output.stable_score_dict,
        input_shape=input_shape,
        flops_reduction=float(egro_config.get("flops_reduction", 0.5)),
        min_keep_ratio=float(egro_config.get("min_keep_ratio", 0.2)),
        safety_beta=float(egro_config.get("safety_beta", 0.0)),
        eta_g=float(egro_config.get("eta_g", 0.05)),
        delta=float(egro_config.get("delta", 1e-12)),
    )
    egro_output = egro.run()
    run_metrics = {
        **base_metrics,
        **tcsm_output.metrics,
        **egro_output.metrics,
        "layer_keep_ratios": egro_output.layer_keep_ratios,
    }
    sparse_method = EGROTrainingMethod(
        model=model,
        groups=egro_output.groups,
        group_mask=egro_output.group_mask,
        group_omega=egro_output.group_omega,
        lambda0=float(egro_config.get("lambda0", config["method"]["lambda0"])),
        delta=float(egro_config.get("delta", 1e-12)),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    max_train_batches, max_val_batches = bounded_batches(config)
    run_name = build_run_name(
        method="chapter5_egro",
        scorer=experiment["scorer"],
        model=experiment["model"],
        sparsity=experiment["sparsity"],
        flops_reduction=experiment["flops_reduction"],
    )
    logger = create_logger(args, experiment, config, run_name)
    fit_result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(config["training"].get("epochs", 1)),
        method=sparse_method,
        scheduler=scheduler,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        eval_interval=int(config["training"].get("eval_interval", 1)),
        checkpoint_interval=int(config["training"].get("checkpoint_interval", 0)),
        logger=logger,
        base_metrics=run_metrics,
        log_batches=bool(config["training"].get("log_batches", False)),
        log_interval=int(config["training"].get("log_interval", 1)),
        extra_epoch_metrics=lambda _epoch, _total: egro_output.metrics,
    )
    export_metadata: dict[str, Any] = {"status": "future"}
    export_metrics = {"loss": 0.0, "accuracy": 0.0}
    if experiment["model"] in {"vgg11_bn", "vgg16_bn"}:
        artifact = export_vgg_slim_artifact(
            model=model,
            group_mask=egro_output.group_mask,
            groups=egro_output.groups,
            num_classes=int(config["model"]["num_classes"]),
            input_shape=input_shape,
        )
        export_metadata = artifact.metadata
        export_metrics = evaluate(artifact.model, val_loader, criterion, device, max_batches=max_val_batches)
    summary = {
        **experiment,
        **dataset_record_fields(config),
        "run_dir": logger.run_dir,
        **run_metrics,
        "export_metadata": export_metadata,
        "epochs_ran": fit_result.epochs,
        "train_loss": fit_result.final_train["loss"],
        "train_acc": fit_result.final_train["accuracy"],
        "val_loss": export_metrics["loss"],
        "val_acc": export_metrics["accuracy"],
    }
    logger.write_summary(summary)
    return summary


def execute_experiment(args: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    if experiment["status"] != "supported":
        return {**experiment, "skipped": True, "reason": experiment["status"]}
    if experiment["chapter"] == "chapter3":
        return run_chapter3(args, experiment)
    if experiment["chapter"] == "chapter4":
        return run_chapter4(args, experiment)
    if experiment["chapter"] == "chapter5":
        return run_chapter5(args, experiment)
    raise ValueError(f"Unsupported chapter: {experiment['chapter']}")


def print_dry_run(experiments: list[dict[str, Any]]) -> None:
    for experiment in experiments:
        print(json.dumps(experiment, ensure_ascii=False, sort_keys=True))
    print(f"dry_run_count: {len(experiments)}")


def run_chapter(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    experiments = build_experiments(config, args.profile, args.chapter)
    experiments = apply_cli_overrides(experiments, config, args)
    if args.max_runs is not None:
        experiments = experiments[: int(args.max_runs)]
    if args.dry_run or args.profile == "full":
        print_dry_run(experiments)
        if args.profile == "full" and not args.dry_run:
            print("full_profile_is_dry_run_only: true")
        return

    summaries: list[dict[str, Any]] = []
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        summary = execute_experiment(args, experiment)
        summaries.append(summary)
        append_jsonl(assets_dir / f"{args.chapter}_summary.jsonl", summary)

    flattened = [flatten_metrics(summary) for summary in summaries]
    summary_csv = assets_dir / f"{args.chapter}_summary.csv"
    write_csv(summary_csv, flattened)
    print(f"records_written: {len(summaries)}")
    print(f"summary_csv: {summary_csv}")


def collect_assets(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted((args.output_dir / "assets").glob("*_summary.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(flatten_metrics(json.loads(stripped)))
    if not records:
        raise ValueError(f"No thesis asset summaries found under {args.output_dir / 'assets'}")
    output = args.output_dir / "assets" / "thesis_summary.csv"
    write_csv(output, records)
    print(f"records_written: {len(records)}")
    print(f"summary_csv: {output}")


def plot_assets(args: argparse.Namespace) -> None:
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "future",
        "message": "Plotting entry point is reserved; CSV/JSONL assets are already produced.",
    }
    with (assets_dir / "plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"plot_manifest: {assets_dir / 'plot_manifest.json'}")


def default_config_for_chapter(chapter: str) -> Path:
    return PROJECT_ROOT / "configs" / "thesis" / f"{chapter}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis experiment capability checks.")
    parser.add_argument("chapter", choices=["chapter3", "chapter4", "chapter5", "collect", "plot"])
    parser.add_argument("--profile", default="quick", choices=["quick", "mini-real", "full"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thesis"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer.lr.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataset.batch_size.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name.")
    parser.add_argument("--sparsity", type=float, default=None, help="Override pruning sparsity.")
    parser.add_argument("--scorer", type=str, default=None, help="Override pruning scorer.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--method", type=str, default=None, help="Override method variant.")
    parser.add_argument("--flops-reduction", type=float, default=None, help="Override EGRO FLOPs reduction.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches per epoch.")
    parser.add_argument("--log-batches", action="store_true", help="Write and print batch-level train logs.")
    parser.add_argument("--log-interval", type=int, default=None, help="Batch log interval.")
    args = parser.parse_args()

    if args.chapter in {"collect", "plot"}:
        collect_assets(args) if args.chapter == "collect" else plot_assets(args)
        return

    if args.config is None:
        args.config = default_config_for_chapter(args.chapter)
    run_chapter(args)


if __name__ == "__main__":
    main()
