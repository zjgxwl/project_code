# Sparse Subnet Optimization Research Skeleton

This repository contains a minimal PyTorch project skeleton for future implementations of:

- Chapter 3: TSPR, training-signal-preserving sparse subnet parameter optimization.
- Chapter 4: TCSM, topology-consistency modeling for sparse subnet structure optimization.
- Chapter 5: EGRO, efficient sparse subnet computation through structural reorganization.

The current stage intentionally does not implement the full algorithms. It only provides a clean package layout, smoke-testable training loop, and placeholder interfaces.

## Conda Environment

Run from the `project_code` directory:

```powershell
conda env create -f environment.yml
conda activate sso
```

Install PyTorch and torchvision inside the activated `sso` environment according to your machine:

- For CUDA experiments, install the torch/torchvision builds that match your local CUDA version.
- For CPU-only smoke tests, install the CPU builds of torch/torchvision.

After PyTorch is installed, install this project in editable mode:

```powershell
python -m pip install -e .
```

## Tests

```powershell
pytest -q
```

## Result Logging

Experiment scripts keep their default smoke-test behavior unless
`--save-metrics` is passed. Saved metrics are written under `outputs/runs/` as
one JSON file per run plus an appended `metrics.jsonl` file.

```powershell
python scripts/train_tspr.py --config configs/debug.yaml --sparsity 0.9 --lambda0 1e-4 --scorer snip --save-metrics
python scripts/debug_tcsm.py --config configs/debug.yaml --scorer snip --sparsity 0.9 --save-metrics
python scripts/export_egro_vgg.py --config configs/debug.yaml --model vgg11_bn --scorer snip --flops-reduction 0.5 --save-metrics
python scripts/collect_results.py --input outputs/runs/metrics.jsonl --output outputs/paper_assets/results_summary.csv
```

The `outputs/` directory is a runtime artifact location and is not intended for
source control by default. `outputs/paper_assets/` is only an entry point for
paper result assets; it does not automatically modify any LaTeX thesis files.

## Smoke Test

The debug configuration uses CPU-friendly settings and `FakeData` by default, so it does not require a GPU or dataset download.

```powershell
python scripts/debug_pruning.py --config configs/debug.yaml --sparsity 0.9 --scorer magnitude
python scripts/debug_pruning.py --config configs/debug.yaml --sparsity 0.9 --scorer snip
python scripts/debug_pruning.py --config configs/debug.yaml --sparsity 0.9 --scorer synflow
python scripts/debug_pruning.py --config configs/debug.yaml --sparsity 0.9 --scorer grasp
python scripts/train_sparse_retrain.py --config configs/debug.yaml --sparsity 0.9
python scripts/train_sparse_retrain.py --config configs/debug.yaml --sparsity 0.9 --scorer snip
python scripts/train_sparse_retrain.py --config configs/debug.yaml --sparsity 0.9 --scorer grasp
python scripts/train_tspr.py --config configs/debug.yaml --sparsity 0.9 --lambda0 1e-4
python scripts/train_tspr.py --config configs/debug.yaml --sparsity 0.9 --lambda0 1e-4 --scorer snip
python scripts/train_tspr.py --config configs/debug.yaml --sparsity 0.9 --lambda0 1e-4 --scorer grasp
python scripts/debug_tcsm.py --config configs/debug.yaml --scorer snip --sparsity 0.9
python scripts/train_tcsm_tspr.py --config configs/debug.yaml --scorer snip --sparsity 0.9 --lambda0 1e-4
python scripts/debug_egro.py --config configs/debug.yaml --scorer snip --flops-reduction 0.5
python scripts/train_egro.py --config configs/debug.yaml --scorer snip --flops-reduction 0.5 --lambda0 1e-4
python scripts/export_egro_vgg.py --config configs/debug.yaml --model vgg11_bn --scorer snip --flops-reduction 0.5
python scripts/train_baseline.py --config configs/debug.yaml
```

The current GraSP scorer is a minimal second-order engineering loop for smoke
tests and method comparison plumbing, not a full paper-level GraSP reproduction.

The current TCSM path is also a minimal engineering loop. It uses FakeData and
small calibration batches to simulate lightweight observations, and does not
include data condensation, random real subsets, full-data calibration, or a
full Chapter 4 experimental reproduction.

The current EGRO path is a Stage-1 minimal engineering loop. It only implements
Conv2d output-channel group modeling, group-level score aggregation, Conv FLOPs
estimation, and group mask generation. It does not perform real structured
model export, BatchNorm synchronization, next-layer input-channel pruning,
residual-branch synchronization, dependency-closure rewriting, or group-level
partial regularization training. The `safety_beta` option is reserved for later
extensions and is not used by the Stage-1 safety bound.

The current EGRO Stage-2 path only implements group-level partial
regularization training and same-shape group-masked state dict export for smoke
evaluation. It still does not perform slim model export, dependency-consistent
rewriting, BatchNorm synchronization, next-layer input-channel pruning, or
residual-branch synchronization, and it does not represent real deployment
speedup.

The current EGRO Stage-3a path only supports the project-local VGG-style serial
CNN for real slim model export. The exported model has smaller parameter tensor
shapes, but this path does not support ResNet, shortcut branches, residual
connections, or a generic dependency graph, and it does not run real latency
benchmarks.
