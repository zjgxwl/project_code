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
python scripts/train_baseline.py --config configs/debug.yaml
```

The current GraSP scorer is a minimal second-order engineering loop for smoke
tests and method comparison plumbing, not a full paper-level GraSP reproduction.

The current TCSM path is also a minimal engineering loop. It uses FakeData and
small calibration batches to simulate lightweight observations, and does not
include data condensation, random real subsets, full-data calibration, or a
full Chapter 4 experimental reproduction.
