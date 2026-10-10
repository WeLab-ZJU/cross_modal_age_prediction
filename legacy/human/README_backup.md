# Cross-Modal Biological Age Prediction

Reference implementation for a multi-modal pipeline that estimates biological
age from three complementary modalities:

1. **Eye imaging** — paired fundus and OCTA images, each processed by an
   HCF-pretrained Vision Transformer (ViT-Base/16, ViT-Large/16 or
   ViT-Huge/14).
2. **Gait** — low-dimensional gait descriptors scored with a classical ML
   classifier and aggregated across sequences per subject.
3. **Behavior** — summaries of open-field-test (OFT) pose-tracking, scored
   with a classical ML classifier per subject.

The three modality vectors are L2-normalised and linearly fused (optionally
with fixed weights) into a single per-subject prediction. A downstream
module then converts the fused class-probability vector into a continuous
biological-age score.

## Repository layout

```
cross_modal_age_prediction/
├── README.md
├── requirements.txt
├── models_vit.py              # ViT backbones (Base / Large / Huge)
├── train_eye.py               # HCF fine-tuning on eye imaging
├── train_gait.py              # Classical ML on gait descriptors
├── train_behavior.py          # Classical ML on OFT behavior descriptors
├── fuse_modalities.py         # Tri-modal fusion + evaluation + Excel report
├── predict_continuous_age.py  # Continuous biological-age score from fusion
├── scripts/
│   ├── train_eye_base.sh      # ViT-Base/16 training template
│   ├── train_eye_large.sh     # ViT-Large/16 training template
│   └── train_eye_huge.sh      # ViT-Huge/14 training template
├── demo/
│   ├── run_demo_gait.sh
│   ├── run_demo_behavior.sh
│   ├── run_demo_continuous_age.sh
│   └── run_demo_fusion.sh     # Template – requires real checkpoints/images
└── data_examples/             # Small synthetic CSVs used by the demos
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 and CUDA 12.1 were used during development. For CPU-only
execution of the classical-ML demos, PyTorch is not strictly required.

The HCF fine-tuning code (`train_eye.py`) additionally depends on the
`util/` helpers and `engine_finetune.py` from the official HCF repository
(https://github.com/facebookresearch/HCF). Place those files next to
`train_eye.py` before training.

## Quickstart (three self-contained demos)

The CSVs in `data_examples/` are small synthetic datasets so the three
demos below can be run without any real data or GPU.

```bash
# 1. Gait modality: train and evaluate classical ML classifiers
bash demo/run_demo_gait.sh

# 2. Behavior modality: train and evaluate classical ML classifiers
bash demo/run_demo_behavior.sh

# 3. Turn a fused-feature CSV into a continuous biological-age score
bash demo/run_demo_continuous_age.sh
```

Each demo prints per-class and macro-averaged metrics; the continuous-age
demo additionally writes a scored CSV and a PNG figure under
`outputs/demo_continuous_age/`.

## Training the eye modality

`train_eye.py` fine-tunes an HCF-pretrained ViT on an ImageFolder-style
directory with `train/`, `val/` and `test/` subfolders, one per cohort
(`8W`, `3M`, `12M`, `18M`).

| Backbone        | Script                        | Input size | Epochs |
|-----------------|-------------------------------|------------|--------|
| ViT-Base/16     | `scripts/train_eye_base.sh`   | 196        | 1100   |
| ViT-Large/16    | `scripts/train_eye_large.sh`  | 256        | 700    |
| ViT-Huge/14     | `scripts/train_eye_huge.sh`   | 256        | 500    |

Edit the `DATA_ROOT`, `HCF_CKPT` and `OUTPUT_ROOT` variables at the top of
each script, then run e.g.:

```bash
bash scripts/train_eye_base.sh
```

The script launches `torch.distributed.run` over two GPUs; adjust
`--nproc_per_node`, `--batch_size` and `--world_size` to fit your hardware.
The best checkpoint (by validation AUC) is written to
`${OUTPUT_ROOT}/${TASK}/checkpoint-best.pth`.

## Training the gait modality — `train_gait.py`

Per-subject gait descriptors are stored in a CSV containing an `id` column,
one or more `featureN` columns, and a `label` column with integer class
indices (0..3 → 8W, 3M, 12M, 18M). Each subject may contribute multiple
sequences; per-sequence class probabilities are summed per subject before
argmax.

```bash
python train_gait.py \
    --train_csv data_examples/gait_train.csv \
    --test_csv  data_examples/gait_test.csv \
    --feature_cols feature1 feature2 \
    --model_dir  outputs/gait_model \
    --classifiers ExtraTrees RandomForest
```

Supported classifiers: `baseline`, `AdaBoost`, `Bagging`, `DecisionTree`,
`ExtraTree`, `ExtraTrees`, `GaussianProcess`, `GradientBoosting`,
`KNeighbors`, `RandomForest`, `MLP`, `LM`.

For each classifier the script saves three artefacts to `--model_dir`:
`model-<name>.pkl`, `imputer-<name>.pkl`, `scaler-<name>.pkl`.

## Training the behavior modality — `train_behavior.py`

Per-subject OFT descriptors are stored in a CSV with `id`, `feature1`..
`featureN`, and `label` columns. Unlike gait, the behavior modality uses
one row per subject; no aggregation is performed.

```bash
python train_behavior.py \
    --train_csv data_examples/behavior_train.csv \
    --test_csv  data_examples/behavior_test.csv \
    --n_features 28 \
    --model_dir outputs/behavior_model \
    --classifiers MLP LM ExtraTrees
```

Additional classifier available (beyond the gait list):
`HistGradientBoosting`.

## Fusing the three modalities — `fuse_modalities.py`

Once you have (a) two fine-tuned eye ViT checkpoints, (b) a saved gait
classifier and (c) a saved behavior classifier, `fuse_modalities.py`
runs each modality on the test set, aligns per-subject features, and
evaluates seven fusion variants:

| Key                                   | Description                             |
|---------------------------------------|-----------------------------------------|
| `single_feature_predict_eye_A`        | Eye sub-modality A only                 |
| `single_feature_predict_eye_B`        | Eye sub-modality B only                 |
| `cross_feature_predict_eye`           | Fuse both eye sub-modalities            |
| `cross_feature_predict_eye_gait`      | Eye ⊕ Gait                              |
| `cross_feature_predict_eye_behavior`  | Eye ⊕ Behavior                          |
| `cross_feature_predict_gait_behavior` | Gait ⊕ Behavior                         |
| `cross_feature_predict_all`           | Full tri-modal fusion                   |

For every variant the script:

- saves a per-subject fused-feature CSV
  (`<variant>_features_<arch>_<imgsize>_<epochs>.csv`);
- computes overall accuracy / precision / recall / F1 and per-class
  one-vs-rest AUC;
- appends a row to a formatted Excel summary.

Use `demo/run_demo_fusion.sh` as a template and edit the paths at the top.
A minimal CLI invocation:

```bash
python fuse_modalities.py \
    --eye_arch vit_base_patch16 \
    --eye_ckpt_a     /path/to/fundus/checkpoint-best.pth \
    --eye_ckpt_b     /path/to/octa/checkpoint-best.pth \
    --eye_test_root_a /path/to/fundus/test \
    --eye_test_root_b /path/to/octa/test \
    --eye_img_size 256 --eye_epochs 700 --eye_is_augment \
    --behavior_test_csv /path/to/behavior_test.csv \
    --behavior_model_dir outputs/behavior_model \
    --behavior_classifier ExtraTrees \
    --gait_test_csv /path/to/gait_test.csv \
    --gait_model_dir outputs/gait_model \
    --gait_classifier ExtraTrees \
    --gait_feature_cols feature1 feature2 \
    --subject_order S000 S001 S002 S003 ... \
    --output_feature_dir outputs/fused_features \
    --output_report_xlsx outputs/reports/fusion_report.xlsx \
    --use_fixed_weights
```

The `--subject_order` list defines the canonical ordering used for
aligning the three modalities. It must be a permutation of the subject
IDs present in the test CSVs / eye test folders.

## Continuous biological-age score — `predict_continuous_age.py`

Given a fused-feature CSV with columns `class_0..class_3` and a `label`
column (as produced by `fuse_modalities.py`), this script computes a
per-subject continuous age score `S_age`:

\[
S_{\mathrm{age}} = A[i] + \sum_{j \ne i} \frac{p_j}{\sum_k p_k} \cdot c_j
\]

where `i` is the true cohort, `A[i]` is the cohort anchor age (in months),
and `c_j` are fixed age-gap coefficients. Running the demo produces both
a scored CSV and a bar chart with per-cohort mean and variance:

```bash
python predict_continuous_age.py \
    --input_csv  data_examples/fused_features_example.csv \
    --output_dir outputs/continuous_age
```

## Reproducibility notes

- All shell scripts accept environment-variable-style overrides at the top.
- Every classical-ML classifier is seeded and the behavior/gait demos
  produce deterministic output given fixed CSVs.
- The eye-modality training uses a distributed launcher; results will vary
  slightly with a different number of GPUs or batch size.

## Anonymisation

This repository does not contain identifiers, full names, institutional
paths, or real subject IDs from the original experiments. Placeholder
identifiers in `data_examples/` (`S000`, `B000`, ...) are synthetic.

## License

