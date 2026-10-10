# Cross-Modal Age Prediction — Human & Mouse

Training and evaluation workflows for human eye imaging and mouse eye,
gait, and behavior modalities, consolidated into a single repository.

This English-language edition is self-contained: use
`cross_modal_age_prediction_en/` as the repository root. It does not depend
on the neighboring repository or the original `codes/` and `codes_mouse/`
directories.

This is a source-code consolidation, **not a newly validated reproduction
of the experiments**. Translation changes documentation, comments, and one
error message only; model logic, hyperparameters, and evaluation formulas
are unchanged from the consolidated source.

## 1. Repository layout

```text
cross_modal_age_prediction_en/
├── README.md
├── run.py                         # Unified workflow launcher
├── requirements*.txt              # Full, tabular-only, and visualization dependencies
├── human/                         # Python code from the original codes directory
│   ├── train_eye.py               # Single-modality eye training; --eval for testing
│   ├── A_train_eye.py             # Original duplicate training entry point
│   ├── fuse_oct_octa_human.py      # OCT + OCTA fusion and classification evaluation
│   ├── fuse_oct_octa_human2.py     # Fusion, classification, and age/MAE outputs
│   ├── visualize_all_best_models.py
│   ├── models_vit.py
│   ├── engine_finetune.py
│   └── util/
├── mouse/                         # Python code from the original codes_mouse directory
│   ├── train_eye.py               # Eye training and testing
│   ├── train_gait.py              # Gait training and evaluation
│   ├── train_behavior.py          # Behavior training and evaluation
│   ├── fuse_modalities.py         # Eye + gait + behavior fusion evaluation
│   ├── predict_continuous_age.py   # Age score requiring true cohort labels
│   ├── _generate_demo_data.py
│   ├── models_vit.py
│   ├── engine_finetune.py         # Missing dependency supplied from the human code
│   ├── util/
│   └── demo/                     # Three demos with automatic synthetic data generation
├── scripts/                       # Configurable training and evaluation launchers
├── legacy/                        # Historical documentation and experimental scripts
├── docs/
│   ├── DATA_FORMAT.md
│   ├── MIGRATION.md
│   ├── VALIDATION.md
│   └── source_manifest.json       # Original-to-packaged file mappings and SHA-256 hashes
└── tests/                         # Packaging checks without ML dependencies
```

The two copies of `models_vit.py` and `util/` are intentionally kept
separate to minimize changes to existing experiments. Each species uses
its own local imports. The unified launcher runs each workflow in a
separate process to prevent collisions between identically named modules.

`human/fuse_modalities.py` and `human/_generate_demo_data.py` were already
mouse-related duplicates in the original human source directory. They are
retained for completeness; **use `mouse/` for mouse multimodal workflows**.

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

The workflows use Python 3.10+ syntax. The dependency lists were inferred
from imports; **the PyTorch / torchvision / timm combination has not been
pinned or validated**. These files are not reproducibility lock files.
For formal experiments, start from your previously working environment
and export its exact dependency versions separately.

The original eye-training loop includes CUDA-specific calls. CPU/MPS
training support was not added during packaging. Supply your own real
data and model weights, and load only trusted checkpoint and joblib files.

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
`python run.py <species> <workflow> --help` to inspect the original arguments.

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
`NB_CLASSES`, and `PYTHON`. Additional original Python arguments may be
appended to the command.

The new defaults are configurable examples, **not universal replacements
for the original experimental hyperparameters**. Historical settings are
preserved under `legacy/`.

Input dimensions, classifier label order, and checkpoints must match.
Even with `--eval`, the original script constructs the `train/`, `val/`,
and `test/` datasets; all three must exist.

The best checkpoint is saved as
`${OUTPUT_ROOT}/${TASK}/checkpoint-best.pth`. The original engine selects
it using `(F1 + ROC_AUC + Kappa) / 3`, **not AUC alone**, despite the
wording in the historical README.

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
models, and matching test subjects. Review the original alignment
limitations in `docs/MIGRATION.md` before use.

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

The original fusion code selects devices automatically: CPU when CUDA is
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

## 8. Validation and publication checklist

```bash
# Standard library only: no training, dataset, or GPU required
python -B -m unittest discover -s tests -v
```

See `docs/VALIDATION.md` for the validation scope and `docs/MIGRATION.md`
for migration details and known limitations. No real data, weights,
training outputs, or `.git` directory are included. Nothing has been
pushed to a remote repository.

Before publication:

1. Use this directory itself as the repository root, not its parent
   `outputs/` directory.
2. Review `.gitignore`, `git status`, and staged files to avoid publishing
   images, subject information, feature tables, checkpoints, spreadsheets,
   or logs. Adjust image exclusions deliberately if adding public figures.
3. Verify dependency versions, model forward passes, class mappings, and
   subject alignment in the actual training environment before claiming
   reproducibility.
4. Confirm author permissions, third-party notices, paper citations, and
   checkpoint redistribution rights against the original sources.
   **No license was selected and no unverified citation was added during
   packaging or translation.**
