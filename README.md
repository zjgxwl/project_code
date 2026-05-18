# Sparse Subnet Optimization Research Skeleton

This repository contains a minimal PyTorch project skeleton for future implementations of:

- Chapter 3: TSPR, training-signal-preserving sparse subnet parameter optimization.
- Chapter 4: TCSM, topology-consistency modeling for sparse subnet structure optimization.
- Chapter 5: EGRO, efficient sparse subnet computation through structural reorganization.

The current stage intentionally does not implement the full algorithms. It only provides a clean package layout, smoke-testable training loop, and placeholder interfaces.

## Setup

Run from the `project_code` directory:

```powershell
pip install -e .
```

## Tests

```powershell
pytest -q
```

## Smoke Test

The debug configuration uses CPU-friendly settings and `FakeData` by default, so it does not require a GPU or dataset download.

```powershell
python scripts/train_baseline.py --config configs/debug.yaml
```
