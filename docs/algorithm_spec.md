# Algorithm Engineering Specification

This document defines engineering interfaces for future algorithm work. It is not thesis prose and does not claim that TSPR, TCSM, or EGRO are implemented.

## Chapter 3: TSPR

### Inputs

- Dense or sparse model.
- Training batch, labels, optimizer state, and pruning masks.
- Configuration for sparsity target, training-signal preservation terms, and schedules.

### Outputs

- Updated sparse subnet parameters.
- Metrics for task loss, preservation loss, sparsity, and mask state.
- Optional checkpoints for later comparison.

### Core interfaces

- `BaseSparseMethod.setup()`
- `BaseSparseMethod.step(...)`
- `TSPRMethod(BaseSparseMethod)`

### Expected files

- `src/sso/methods/tspr.py`
- `src/sso/pruning/`
- `src/sso/metrics/`
- `configs/`

### TODO

- Define the training-signal preservation objective.
- Add mask-aware parameter update hooks.
- Add logging for preservation metrics and sparsity.
- Add config schema for TSPR-specific hyperparameters.

### Unit test plan

- Verify TSPR class construction and interface compatibility.
- Verify a bounded method step preserves tensor shapes.
- Verify metric outputs contain required keys after implementation.

## Chapter 4: TCSM

### Inputs

- Model topology description and candidate sparse masks.
- Calibration data batches.
- Configuration for topology consistency scoring and mask update rules.

### Outputs

- Optimized sparse subnet structure.
- Topology consistency scores and selected masks.
- Optional mask artifacts for downstream training.

### Core interfaces

- `BaseSparseMethod.setup()`
- `BaseSparseMethod.step(...)`
- `TCSMMethod(BaseSparseMethod)`

### Expected files

- `src/sso/methods/tcsm.py`
- `src/sso/pruning/`
- `src/sso/metrics/`
- `configs/`

### TODO

- Define graph/topology representation for supported models.
- Implement topology consistency scoring.
- Implement mask selection and calibration flow.
- Add config schema for TCSM-specific hyperparameters.

### Unit test plan

- Verify TCSM class construction and interface compatibility.
- Verify topology extraction returns stable node/edge metadata.
- Verify mask selection respects target sparsity after implementation.

## Chapter 5: EGRO

### Inputs

- Sparse subnet model and pruning masks.
- Structural reorganization configuration.
- Target deployment constraints such as FLOPs, latency, or channel grouping.

### Outputs

- Reorganized sparse model or export artifact.
- Efficiency metrics such as parameters, FLOPs, and optional latency.
- Mapping metadata between original and reorganized structures.

### Core interfaces

- `BaseSparseMethod.setup()`
- `BaseSparseMethod.step(...)`
- `EGROMethod(BaseSparseMethod)`

### Expected files

- `src/sso/methods/egro.py`
- `src/sso/export/`
- `src/sso/metrics/`
- `configs/`

### TODO

- Define structural reorganization primitives.
- Implement safe conversion from sparse masks to efficient structures.
- Add export path for reorganized models.
- Add config schema for EGRO-specific hyperparameters.

### Unit test plan

- Verify EGRO class construction and interface compatibility.
- Verify reorganized model preserves output shape after implementation.
- Verify export metadata is generated with required fields.
