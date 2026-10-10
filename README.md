# Cross-Modal Age Prediction — Human & Mouse

统一整理的人类眼球与小鼠眼球、步态、行为模态训练及评测代码。
本目录可单独作为 GitHub 仓库根目录，不依赖上级的 `codes/` 或 `codes_mouse/`。

This repository collects human eye-imaging workflows and mouse eye, gait,
and behavior workflows in one repository, with separate local modules for
each species. It is a source-code consolidation, **not a newly validated
reproduction of the experiments**.

## 1. 目录与入口

```text
cross_modal_age_prediction/
├── README.md
├── run.py                         # 统一命令分发器
├── requirements*.txt              # 完整 / 表格模型 / 可视化依赖清单
├── human/                         # 原 codes 中的 Python 代码
│   ├── train_eye.py               # 眼球单模态训练；--eval 测试
│   ├── A_train_eye.py             # 原有重复训练入口，保留用于追溯
│   ├── fuse_oct_octa_human.py      # OCT + OCTA 融合分类评测
│   ├── fuse_oct_octa_human2.py     # 融合分类评测 + 年龄/MAE 输出
│   ├── visualize_all_best_models.py
│   ├── models_vit.py
│   ├── engine_finetune.py
│   └── util/
├── mouse/                         # 原 codes_mouse 中的 Python 代码
│   ├── train_eye.py               # 眼球训练 / 测试
│   ├── train_gait.py              # 步态训练并评测
│   ├── train_behavior.py          # 行为训练并评测
│   ├── fuse_modalities.py         # 眼球 + 步态 + 行为融合评测
│   ├── predict_continuous_age.py   # 依赖真实年龄组标签的连续年龄分数
│   ├── _generate_demo_data.py
│   ├── models_vit.py
│   ├── engine_finetune.py         # 从 human 补齐的缺失依赖
│   ├── util/
│   └── demo/                     # 3 个自动生成合成数据的演示
├── scripts/                       # 可配置的训练、单模态测试、人类融合入口
├── legacy/                        # 历史 README 和实验脚本，仅供参考
├── docs/
│   ├── DATA_FORMAT.md
│   ├── MIGRATION.md
│   ├── VALIDATION.md
│   └── source_manifest.json       # 原文件 → 新文件及 SHA-256
└── tests/                         # 无 ML 依赖的打包检查
```

为减少迁移对既有实验的影响，没有强行合并两个 `models_vit.py` 和 `util/`；
每个物种仍使用自己的本地导入。统一入口通过独立进程运行，避免同名模块串用。
`human/fuse_modalities.py` 和 `human/_generate_demo_data.py` 是原人类目录中
已有的小鼠代码副本，仅为完整保留而存在；**小鼠多模态工作流请用 `mouse/`**。

## 2. 环境

以下命令均在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate

# 完整的眼球训练 / 融合依赖
python -m pip install -r requirements.txt

# 或：只运行 CPU 表格模型与年龄分数演示
python -m pip install -r requirements-tabular.txt

# 可选：Grad-CAM / EigenCAM
python -m pip install -r requirements-visualization.txt
```

代码使用 Python 3.10+ 语法。依赖清单根据代码导入整理，**没有锁定或验证
PyTorch / torchvision / timm 的组合**，不能视为可复现实验的环境锁文件。
正式训练优先使用你原来能运行的环境，并另行导出版本清单。
原眼球训练循环包含 CUDA 专用调用；本次未进行 CPU/MPS 训练适配。
模型权重和真实数据需要自行提供，仅加载可信来源的 checkpoint / joblib 文件。

## 3. 统一命令

```bash
python run.py --help
python run.py human --help
python run.py mouse --help

# 不加载 ML 依赖，仅打印将执行的命令
python run.py --dry-run human evaluate-eye \
  --data_path data/human/OCT --resume checkpoints/human/OCT/checkpoint-best.pth
```

`run.py` 将工作目录固定到仓库根目录；所有传给它的相对数据和输出路径均
相对于该根目录。直接执行物种脚本时，路径则相对于当前 shell 工作目录。
`mouse/demo/*.sh` 固定在 `mouse/` 内运行，输出到 `mouse/outputs/`。

| Species | Workflow | 实际入口 |
|---|---|---|
| human / mouse | `train-eye` | 对应的 `train_eye.py` |
| human / mouse | `evaluate-eye` | `train_eye.py --eval` |
| human | `fuse-eye` | `fuse_oct_octa_human.py` |
| human | `fuse-eye-age` | `fuse_oct_octa_human2.py` |
| human | `visualize` | `visualize_all_best_models.py` |
| mouse | `train-gait` | `train_gait.py` |
| mouse | `train-behavior` | `train_behavior.py` |
| mouse | `fuse` | `fuse_modalities.py` |
| mouse | `continuous-age` | `predict_continuous_age.py` |
| mouse | `generate-demo` | `_generate_demo_data.py` |

统一入口不改变模型参数；`evaluate-eye` 仅额外添加 `--eval`。
依赖安装完成后，用 `python run.py <species> <workflow> --help` 查看原入口参数。

## 4. 人类 / 小鼠眼球训练与单模态测试

以人类 OCT 为例，OCTA、FP 和小鼠眼球可通过数据路径及 `SPECIES` 切换：

```bash
SPECIES=human \
DATA_ROOT=/absolute/path/to/human/OCT \
HCF_CKPT=/absolute/path/to/pretrained.pth \
MODEL=vit_base_patch16 INPUT_SIZE=256 EPOCHS=500 NPROC=1 \
OUTPUT_ROOT=outputs/human/OCT \
bash scripts/train_eye.sh

SPECIES=human \
DATA_ROOT=/absolute/path/to/human/OCT \
CHECKPOINT=/absolute/path/to/checkpoint-best.pth \
MODEL=vit_base_patch16 INPUT_SIZE=256 \
bash scripts/evaluate_eye.sh
```

小鼠示例：将 `SPECIES=mouse`，并换成小鼠数据和预训练权重。
训练脚本支持 `NPROC`、`BATCH_SIZE`、`MASTER_PORT`、`TASK`、`BLR`、
`LAYER_DECAY`、`DROP_PATH`、`NB_CLASSES`、`PYTHON` 等环境变量；
末尾也可追加原 Python 入口参数。默认模板是便于配置的示例，
**不是原实验超参数的统一替代**，历史参数见 `legacy/`。

模型的输入尺寸、分类头类别顺序和 checkpoint 必须匹配。
即使 `--eval`，原脚本也会构建 `train/val/test` 三个数据集，均需存在。
最佳权重位于 `${OUTPUT_ROOT}/${TASK}/checkpoint-best.pth`；
原引擎实际用于选优的是 `(F1 + ROC_AUC + Kappa) / 3`，
不是原 README 所称的单独 AUC。

## 5. 人类 OCT / OCTA 融合评测

```bash
OCT_CKPT=/absolute/path/to/oct/checkpoint-best.pth \
OCTA_CKPT=/absolute/path/to/octa/checkpoint-best.pth \
OCT_TEST_ROOT=/absolute/path/to/OCT/test \
OCTA_TEST_ROOT=/absolute/path/to/OCTA/test \
OCT_IMG_SIZE=256 OCTA_IMG_SIZE=256 \
bash scripts/fuse_human_eye.sh
```

默认调用 `fuse-eye-age`，且两路都用 `cuda:0`；
可设 `OCTA_DEVICE=cuda:1`，或将两路 device 都设为 `cpu`。
仅进行旧版分类融合可设 `WORKFLOW=fuse-eye`。
不会默认覆盖已有结果；确需覆盖时显式在末尾添加 `--overwrite`。
文件命名、类别顺序和真实年龄提取要求见 `docs/DATA_FORMAT.md`。

## 6. 小鼠步态、行为及连续年龄分数

安装表格模型依赖后：

```bash
bash mouse/demo/run_demo_gait.sh
bash mouse/demo/run_demo_behavior.sh
bash mouse/demo/run_demo_continuous_age.sh
```

脚本会按需在 `mouse/data_examples/` 生成随机合成 CSV，不包含实验数据。
三个演示互相独立；生成的步态 ID 与行为 ID 不同，
**不能直接用于三模态融合**。

真实数据训练：

```bash
python run.py mouse train-gait \
  --train_csv data/mouse/gait_train.csv --test_csv data/mouse/gait_test.csv \
  --feature_cols feature1 feature2 --classifiers ExtraTrees \
  --model_dir outputs/mouse/gait_model

python run.py mouse train-behavior \
  --train_csv data/mouse/behavior_train.csv --test_csv data/mouse/behavior_test.csv \
  --n_features 28 --classifiers ExtraTrees \
  --model_dir outputs/mouse/behavior_model
```

两个入口均执行训练并评测；各分类器保存模型、imputer 和 scaler 三个 `.pkl`。
不应把重跑训练脚本当作“仅评测已有模型”的命令。

## 7. 小鼠三模态融合

必须有两路眼球权重与图像、已训练的步态/行为模型、相同测试个体，
并仔细确认 `docs/MIGRATION.md` 中的原有对齐限制：

```bash
python run.py mouse fuse \
  --eye_arch vit_base_patch16 --eye_img_size 256 --eye_epochs 500 \
  --eye_ckpt_a checkpoints/mouse/eye_a/checkpoint-best.pth \
  --eye_ckpt_b checkpoints/mouse/eye_b/checkpoint-best.pth \
  --eye_test_root_a data/mouse/eye_a/test \
  --eye_test_root_b data/mouse/eye_b/test \
  --behavior_test_csv data/mouse/behavior_test.csv \
  --behavior_model_dir outputs/mouse/behavior_model \
  --behavior_classifier ExtraTrees --behavior_n_features 28 \
  --gait_test_csv data/mouse/gait_test.csv \
  --gait_model_dir outputs/mouse/gait_model \
  --gait_classifier ExtraTrees --gait_feature_cols feature1 feature2 \
  --subject_order S001 S002 S003 S004 \
  --output_feature_dir outputs/mouse/fused_features \
  --output_report_xlsx outputs/mouse/reports/fusion.xlsx \
  --use_fixed_weights
```

`S001 ... S004` 仅是占位 ID，需替换为完整测试个体顺序；
`--eye_epochs` 用于输出文件命名，而非启动训练。
若使用原来的增强文件命名，则按需添加 `--eye_is_augment`。
原融合实现自动选择设备：无 CUDA 时使用 CPU，一块 GPU 时两路共用，
多块 GPU 时分别使用 `cuda:0` 和 `cuda:1`；没有单独的设备 CLI。
连续年龄入口：

```bash
python run.py mouse continuous-age \
  --input_csv /absolute/path/to/fused_features.csv \
  --output_dir outputs/mouse/continuous_age
```

该分数计算使用真实 `label`；不是无需真实年龄组信息的独立年龄回归模型。

## 8. 校验与 GitHub 发布前检查

```bash
# 仅使用 Python 标准库；不启动训练，不需要数据或 GPU
python -B -m unittest discover -s tests -v
```

验证范围见 `docs/VALIDATION.md`，迁移明细与已知限制见 `docs/MIGRATION.md`。
当前没有附带数据、权重、训练结果、`.git` 目录，也未推送远程仓库。

发布前：

1. 上传的是本目录整体，不要把其上级 `outputs/` 当作仓库根目录。
2. 核查 `.gitignore`、`git status` 和待提交文件，避免上传图像、受试者信息、
   特征表、checkpoint、Excel 和日志。新增文档图片应有意识地调整忽略规则。
3. 在实际训练环境验证依赖版本、模型前向、类别与个体对齐后再声明可复现。
4. 根据原代码来源核实授权、第三方 notices、论文引用与权重再分发权限。
   **本次没有代你选择开源许可证或补写未经确认的引用信息。**
