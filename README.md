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
python scripts/debug_pruning.py --config configs/debug.yaml
python scripts/train_baseline.py --config configs/debug.yaml
```
