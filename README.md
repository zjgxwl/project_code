# 稀疏子网优化论文实验工程

本目录是论文实验代码工程，基于 PyTorch 实现并组织以下三章相关能力：

- 第 3 章：`TSPR`，面向训练信号保持的稀疏子网参数优化。
- 第 4 章：`TCSM`，面向拓扑一致性的稀疏子网结构建模。
- 第 5 章：`EGRO`，面向效率的稀疏子网结构重组与结构化导出。

当前工程默认面向真实数据集验证。快速开发和测试也使用真实数据的小 subset。

## 环境配置

请在 `project_code` 目录下执行：

```powershell
conda env create -f environment.yml
conda activate sso
```

然后根据机器环境安装 PyTorch 和 torchvision：

- CUDA 实验：安装与本机 CUDA 版本匹配的 `torch`/`torchvision`。
- CPU 验证：安装 CPU 版本的 `torch`/`torchvision` 即可。

安装 PyTorch 后，将本项目以可编辑模式安装：

```powershell
python -m pip install -e .
```

## 测试

测试依赖真实数据集。推荐先准备 CIFAR-10 到 `outputs/thesis_real_data`，后续单元测试和快速验证会复用该目录。

```powershell
python scripts/run_thesis.py chapter3 --profile quick --output-dir outputs/thesis_real_validation --data-dir outputs/thesis_real_data --download --max-runs 1 --epochs 1 --max-train-batches 1 --max-val-batches 1
pytest -q
```

## 统一论文入口

推荐使用 `scripts/run_thesis.py` 作为论文实验统一入口。它支持 `chapter3`、`chapter4`、`chapter5`、`collect`、`plot` 子命令，并通过 profile 控制实验规模：

- `quick`：真实 CIFAR-10 小 subset，用于快速开发检查。
- `mini-real`：真实 CIFAR-10 小 subset，用于三章代表性链路验收。
- `full`：展开正式论文矩阵，默认只建议 dry-run 或限制 `--max-runs`，避免误跑完整 200 epoch 多 seed 实验。

```powershell
python scripts/run_thesis.py chapter3 --profile quick --max-runs 1 --download
python scripts/run_thesis.py chapter4 --profile quick --max-runs 1
python scripts/run_thesis.py chapter5 --profile quick --max-runs 1
python scripts/run_thesis.py chapter3 --profile mini-real --max-runs 1 --download
python scripts/run_thesis.py chapter5 --profile full --max-runs 5
python scripts/run_thesis.py collect --output-dir outputs/thesis
```

统一入口支持传统训练脚本风格的命令行覆盖，覆盖优先级高于 YAML profile，最终配置仍会写入每个 run 目录的 `config.yaml`：

```powershell
python scripts/run_thesis.py chapter3 --profile quick --epochs 100 --lr 0.1 --batch-size 128 --model resnet18 --dataset cifar10 --sparsity 0.95 --scorer snip
python scripts/run_thesis.py chapter4 --profile mini-real --max-runs 1 --epochs 1 --max-train-batches 1 --max-val-batches 1 --log-batches --log-interval 1
python scripts/run_thesis.py chapter5 --profile mini-real --model vgg16_bn --flops-reduction 0.5 --log-batches
```

每个 run 会写入标准目录结构：

- `config.yaml`：最终实验配置快照。
- `command.txt`：启动命令。
- `env.json`：Python、PyTorch、CUDA 等环境信息。
- `metrics_epoch.jsonl`：逐 epoch 指标。
- `metrics_batch.jsonl`：可选逐 batch 指标，使用 `--log-batches` 开启。
- `summary.json`：单次 run 汇总。
- `checkpoints/`：可选 checkpoint。
- `artifacts/`：结构化导出、分析文件等附加产物。

epoch 和 batch 日志包含训练/测试 loss、accuracy、学习率、耗时、目标稀疏率、实际稀疏率、保留/剪除参数量、参数下降比例、模型参数量、Conv FLOPs，以及 TSPR/TCSM/EGRO 的关键方法指标。控制台也会打印类似传统训练脚本的格式：

```text
Epoch [1/100] Batch [20/391] train_loss=... train_acc=... batch_loss=... batch_acc=... lr=...
Epoch [1/100] lr=... train_loss=... train_acc=... test_loss=... test_acc=... model=... dataset=... scorer=... sparsity=...
```

外部或后续才实现的基线结果可以导入统一资产格式：

```powershell
python scripts/import_thesis_results.py --input path/to/results.csv --chapter chapter3 --method tost --source paper_reference
```

## 论文能力状态

当前工程已实现论文方法链路的核心能力，但仍按“能力验证优先、完整复现后置”的原则组织：

- 第 3 章：支持 `standard`、`tspr` 小规模训练，支持 Magnitude/SNIP/SynFlow/GraSP/EP 等评分接口。`ToST`、`RePaIR`、`RigL` 不在本工程实现范围内，可通过外部结果导入。
- 第 4 章：支持 `base_only`、`tcsm`、`random_subset`、`full_data` calibration mode，包含 score rank correlation、mask Jaccard、稳定权重等指标。
- 第 5 章：支持 EGRO 的组级评分、FLOPs 估计、组掩码、部分正则化训练和 VGG 风格串行 CNN 的 slim model 导出。ResNet/MobileNet 的依赖闭包结构化导出仍标注为后续扩展。
- 模型：已支持 `resnet18`、`resnet34`、`vgg11_bn`、`vgg16_bn`、`mobilenetv2`；`vit_tiny` 保留为后续扩展。
- 数据：支持 CIFAR-10、CIFAR-100、Tiny-ImageNet，以及 TCSM 校准用的公开 MTT 蒸馏张量准备路径。

## 结果日志

旧训练脚本传入 `--save-metrics` 后会把指标写入 `outputs/runs/`，包括每次运行的 JSON 文件和追加式 `metrics.jsonl`。

```powershell
python scripts/train_tspr.py --config configs/debug.yaml --sparsity 0.9 --lambda0 1e-4 --scorer snip --save-metrics
python scripts/debug_tcsm.py --config configs/debug.yaml --scorer snip --sparsity 0.9 --save-metrics
python scripts/export_egro_vgg.py --config configs/debug.yaml --model vgg11_bn --scorer snip --flops-reduction 0.5 --save-metrics
python scripts/collect_results.py --input outputs/runs/metrics.jsonl --output outputs/paper_assets/results_summary.csv
```

`outputs/` 是运行产物目录，默认不应纳入源码管理。`outputs/paper_assets/` 只用于保存论文结果资产，不会自动修改 LaTeX 论文文件。

## 真实数据验证

真实数据验证使用 `--download` 显式下载数据。CIFAR-10/CIFAR-100 由 torchvision 处理；Tiny-ImageNet 可以手动下载 zip 并设置 `dataset.archive_path`，或解压到 `data/tiny-imagenet-200`。

```powershell
python scripts/train_baseline.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data
python scripts/debug_pruning.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --save-metrics
python scripts/train_tspr.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --lambda0 1e-4 --save-metrics
python scripts/debug_tcsm.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --save-metrics
python scripts/export_egro_vgg.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data --model vgg11_bn --scorer snip --flops-reduction 0.5 --save-metrics
```

`data/` 和 `outputs/` 都是运行时产物目录，不建议纳入源码管理。快速验证命令只运行少量 batch，用于验证格式和流程，不代表完整论文实验。

## 快速验证命令

`configs/debug.yaml` 默认使用真实 CIFAR-10 小 subset。首次运行可传入 `--download` 准备数据。

```powershell
python scripts/debug_pruning.py --config configs/debug.yaml --download --data-dir outputs/thesis_real_data --sparsity 0.9 --scorer magnitude
python scripts/debug_pruning.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9 --scorer snip
python scripts/debug_pruning.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9 --scorer synflow
python scripts/debug_pruning.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9 --scorer grasp
python scripts/debug_pruning.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9 --scorer ep
python scripts/train_sparse_retrain.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9
python scripts/train_tspr.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --sparsity 0.9 --lambda0 1e-4 --scorer snip
python scripts/debug_tcsm.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9
python scripts/debug_tcsm.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --calibration-mode base_only
python scripts/debug_tcsm.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --calibration-mode random_subset --stability-repeats 2
python scripts/train_tcsm_tspr.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --sparsity 0.9 --lambda0 1e-4
python scripts/debug_egro.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --flops-reduction 0.5
python scripts/train_egro.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --scorer snip --flops-reduction 0.5 --lambda0 1e-4
python scripts/export_egro_vgg.py --config configs/debug.yaml --data-dir outputs/thesis_real_data --model vgg11_bn --scorer snip --flops-reduction 0.5
python scripts/train_baseline.py --config configs/debug.yaml --data-dir outputs/thesis_real_data
```

## TCSM 蒸馏校准数据

TCSM 支持显式下载公开蒸馏校准张量。当前内置公开来源为 MTT 张量下载：CIFAR-10/CIFAR-100 支持 IPC 1/10/50，Tiny-ImageNet 支持 IPC 1/10。论文第 4 章的 IPC=5 设置通过从公开 IPC=10 张量中每类取前 5 个样本生成，并写入派生 metadata。

```powershell
python scripts/download_calibration_data.py --source mtt --dataset cifar10 --ipc 1 --output-dir data/calibration
python scripts/prepare_thesis_calibration_data.py --output-dir data/calibration
python scripts/debug_tcsm.py --config configs/debug.yaml --scorer snip --sparsity 0.9 --calibration-source mtt --calibration-ipc 1 --calibration-data-dir data/calibration/mtt/cifar10/ipc1
```

除非给 TCSM 脚本传入 `--download-calibration`，或直接运行下载脚本，否则不会自动下载校准数据。`data/` 默认由 git 忽略。

## 方法实现说明

当前 GraSP scorer 是用于快速验证和方法比较管线的最小二阶工程实现，不等同于完整论文级 GraSP 复现。

当前 EP scorer 是轻量 Edge-Popup 风格的 score-logit 校准循环，用于管线接入。它会保持模型权重固定并在评分后恢复模型状态，但不是长程 Edge-Popup 完整训练复现。

`ToST`、`RePaIR`、`RigL` 等外部训练基线不在本工程实现范围内。固定掩码稀疏重训练脚本主要用于检查 TSPR 输入和 mask 流程。

当前 TCSM 路径是第 4 章论文方法的工程链路，支持 `base_only`、`tcsm`、`random_subset`、`full_data` 四种校准模式，以及轻量的 score-rank、mask-Jaccard 稳定性检查。它不实现外部数据凝练算法；凝练或蒸馏数据应离线准备，再通过正常 dataset/dataloader 路径传入。

当前 EGRO Stage-1 实现 Conv2d 输出通道组建模、组级评分聚合、Conv FLOPs 估计和组掩码生成。

当前 EGRO Stage-2 实现组级部分正则化训练，以及 same-shape group-masked state dict 导出，用于快速评估。

当前 EGRO Stage-3a 支持项目本地 VGG 风格串行 CNN 的真实 slim model 导出。导出模型具有更小的参数张量形状，并报告实际参数下降、Conv FLOPs 下降和逐层 keep ratio。ResNet 残差分支同步重写、依赖闭包剪枝、外部 dependency graph 剪枝算法仍未声明为已实现。
