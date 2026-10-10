# Cross-Modal Age Prediction — Human & Mouse

Training and evaluation workflows for human eye imaging and mouse eye,
gait, and behavior modalities.

The `english` branch contains the English documentation; the `chinese`
branch contains the Chinese documentation. Run commands from the repository root.

## 1. Repository layout

```text
cross_modal_age_prediction/
├── README.md
├── run.py                         # Unified workflow launcher
├── requirements*.txt              # Full, tabular-only, and visualization dependencies
├── human/                         # Human eye-imaging workflows
│   ├── train_eye.py               # Single-modality eye training; --eval for testing
│   ├── A_train_eye.py             # Alternative eye-training entry point
│   ├── fuse_oct_octa_human.py      # OCT + OCTA fusion and classification evaluation
│   ├── fuse_oct_octa_human2.py     # Fusion, classification, and age/MAE outputs
│   ├── visualize_all_best_models.py
│   ├── models_vit.py
│   ├── engine_finetune.py
│   └── util/
├── mouse/                         # Mouse multimodal workflows
│   ├── train_eye.py               # Eye training and testing
│   ├── train_gait.py              # Gait training and evaluation
│   ├── train_behavior.py          # Behavior training and evaluation
│   ├── fuse_modalities.py         # Eye + gait + behavior fusion evaluation
│   ├── predict_continuous_age.py   # Age score requiring true cohort labels
│   ├── _generate_demo_data.py
│   ├── models_vit.py
│   ├── engine_finetune.py         # Training and evaluation engine
│   ├── util/
│   └── demo/                     # Three demos with automatic synthetic data generation
├── scripts/                       # Configurable training and evaluation launchers
├── legacy/                        # Historical experiment configuration references
├── docs/
│   ├── DATA_FORMAT.md
│   └── USAGE_NOTES.md
└── tests/                         # Launcher and source-code checks
```

Each species uses its own `models_vit.py` and `util/` modules.
The unified launcher runs each workflow in a
separate process to prevent collisions between identically named modules.

**Use `mouse/` for mouse multimodal workflows**, rather than the duplicate
`fuse_modalities.py` and `_generate_demo_data.py` files under `human/`.

## 2. Environment

Run the following commands from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate

# Full eye-training and fusion dependencies
python -m pip install -r requirements.txt

# Alternatively: CPU-only tabular models and continuous-age demos
python -m pip install -r requirements-tabular.txt

# Optional: Grad-CAM and EigenCAM
python -m pip install -r requirements-visualization.txt
```

Use Python 3.10+ and a compatible PyTorch / torchvision / timm environment.
Dependency versions are not pinned; see `docs/USAGE_NOTES.md` for
model compatibility requirements.

Eye training requires CUDA; CPU/MPS training is not supported by the
current loop. Supply your own data and weights, and load only trusted
checkpoint and joblib files.

## 3. Unified commands

```bash
python run.py --help
python run.py human --help
python run.py mouse --help

# Print the command without importing ML dependencies
python run.py --dry-run human evaluate-eye \
  --data_path data/human/OCT --resume checkpoints/human/OCT/checkpoint-best.pth
```

`run.py` sets the working directory to the repository root. Relative data
and output paths passed to it are resolved from that root. When running a
species script directly, paths are relative to the current shell working
directory. The `mouse/demo/*.sh` scripts run inside `mouse/` and write to
`mouse/outputs/`.

| Species | Workflow | Entry point |
|---|---|---|
| human / mouse | `train-eye` | Corresponding `train_eye.py` |
| human / mouse | `evaluate-eye` | `train_eye.py --eval` |
| human | `fuse-eye` | `fuse_oct_octa_human.py` |
| human | `fuse-eye-age` | `fuse_oct_octa_human2.py` |
| human | `visualize` | `visualize_all_best_models.py` |
| mouse | `train-gait` | `train_gait.py` |
| mouse | `train-behavior` | `train_behavior.py` |
| mouse | `fuse` | `fuse_modalities.py` |
| mouse | `continuous-age` | `predict_continuous_age.py` |
| mouse | `generate-demo` | `_generate_demo_data.py` |

The launcher does not modify model parameters. `evaluate-eye` only adds
`--eval`. After installing the dependencies, use
`python run.py <species> <workflow> --help` to inspect the arguments.

## 4. Human / mouse eye training and single-modality evaluation

The following example uses human OCT. To use OCTA, FP, or mouse eye data,
change the data paths and, where appropriate, `SPECIES`:

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

For mice, set `SPECIES=mouse` and provide the corresponding mouse dataset
and pretrained weights. Supported environment variables include `NPROC`,
`BATCH_SIZE`, `MASTER_PORT`, `TASK`, `BLR`, `LAYER_DECAY`, `DROP_PATH`,
`NB_CLASSES`, and `PYTHON`. Additional Python arguments may be
appended to the command.

Configure hyperparameters for your experiment. Additional historical
configuration references are available under `legacy/`.

Input dimensions, classifier label order, and checkpoints must match.
Even with `--eval`, the script constructs the `train/`, `val/`,
and `test/` datasets; all three must exist.

The best checkpoint is saved as
`${OUTPUT_ROOT}/${TASK}/checkpoint-best.pth`. The engine selects
it using `(F1 + ROC_AUC + Kappa) / 3`.

## 5. Human OCT / OCTA fusion evaluation

```bash
OCT_CKPT=/absolute/path/to/oct/checkpoint-best.pth \
OCTA_CKPT=/absolute/path/to/octa/checkpoint-best.pth \
OCT_TEST_ROOT=/absolute/path/to/OCT/test \
OCTA_TEST_ROOT=/absolute/path/to/OCTA/test \
OCT_IMG_SIZE=256 OCTA_IMG_SIZE=256 \
bash scripts/fuse_human_eye.sh
```

The default workflow is `fuse-eye-age`, with both models on `cuda:0`.
Set `OCTA_DEVICE=cuda:1` to use a second GPU, or set both device variables
to `cpu` for CPU inference. To run the older classification-only fusion
workflow, set `WORKFLOW=fuse-eye`.

Existing outputs are not overwritten by default. Append `--overwrite`
only when replacement is intended. See `docs/DATA_FORMAT.md` for filename,
class-order, and true-age extraction requirements.

## 6. Mouse gait, behavior, and continuous-age scores

After installing the tabular dependencies:

```bash
bash mouse/demo/run_demo_gait.sh
bash mouse/demo/run_demo_behavior.sh
bash mouse/demo/run_demo_continuous_age.sh
```

These scripts generate random synthetic CSV files under
`mouse/data_examples/` when needed. No experimental data are included.
The three demos are independent. Their gait and behavior IDs differ,
so **the generated datasets cannot be used directly for tri-modal fusion**.

Training on real tabular data:

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

Both entry points train and evaluate. Each classifier produces three
`.pkl` files: the model, imputer, and scaler. Rerunning either training
script is not an evaluation-only command for an existing model.

## 7. Mouse tri-modal fusion

You need two eye checkpoints and their images, trained gait and behavior
models, and matching test subjects. See `docs/USAGE_NOTES.md` for
subject-alignment requirements.

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

Replace `S001 ... S004` with the complete ordered list of test subjects.
`--eye_epochs` is used in output filenames; it does not start training.
Add `--eye_is_augment` only when using the expected augmented-image naming
scheme.

The fusion code selects devices automatically: CPU when CUDA is
unavailable, a shared GPU when only one is available, or `cuda:0` and
`cuda:1` when multiple GPUs are available. It has no separate device CLI.

Continuous-age scoring:

```bash
python run.py mouse continuous-age \
  --input_csv /absolute/path/to/fused_features.csv \
  --output_dir outputs/mouse/continuous_age
```

The calculation uses the true `label`. It is not an independent age
regression model that operates without true cohort information.

## 8. Tests

```bash
# Standard library only: no training, dataset, or GPU required
python -B -m unittest discover -s tests -v
```

These tests check syntax, launchers, argument forwarding, and local module
availability. They do not run model training or inference.
