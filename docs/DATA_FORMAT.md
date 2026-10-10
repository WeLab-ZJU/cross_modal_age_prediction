# Data and Label Conventions

Data layouts, class mappings, and subject alignment requirements.

## Eye images

```text
data/<species>/<modality>/
├── train/<class_name>/*.png
├── val/<class_name>/*.png
└── test/<class_name>/*.png
```

- Single-modality training and `--eval` use ImageFolder's class-directory
  ordering. All three splits must contain the same class directories,
  with identical directory-to-integer label mappings.
- The human examples use `1-30`, `30-60`, `60-80`, and `80+`.
  Boundary-age assignment is defined by the dataset.
- Mouse fusion assumes that the eye classifier outputs
  `12M, 18M, 3M, 8W`, then reorders the columns to `8W, 3M, 12M, 18M`.
- Do not mix human checkpoints, classes, or label semantics into mouse
  workflows.
- Use `--resume` to load weights for direct evaluation.
  `--finetune` does not load the evaluation checkpoint in `--eval` mode.

## Human OCT / OCTA fusion

`fuse_oct_octa_human.py` and `fuse_oct_octa_human2.py` scan images by
class and infer subjects and eye sides using `split_subject_side()` and
`group_lr_pairs()`. Check your filenames against these parsers; arbitrary
filenames are not guaranteed to pair correctly.

The newer script uses `extract_real_age_from_path()` and
`extract_real_age_from_files()` to extract ages from paths or filenames.
Missing ages or incompatible naming conventions affect MAE availability.
Use `--class_age_values` to explicitly specify representative ages aligned
with `--class_order`. Inspect a sample of the output pairings and age
fields before relying on the results.

## Mouse tabular data

Gait CSV:

```csv
id,feature1,feature2,label
S001,0.1,0.2,0
S001,0.3,0.4,0
S002,0.5,0.6,1
```

Behavior CSV:

```csv
id,feature1,feature2,label
S001,0.1,0.2,0
S002,0.5,0.6,1
```

- These examples illustrate the format only; they are not sufficient
  datasets for training four-class classifiers.
- Tabular labels follow `0=8W, 1=3M, 2=12M, 3=18M`.
- Gait data may contain multiple sequences per subject. Probabilities are
  aggregated by ID, and every row for a subject must have the same label.
- Behavior data normally contain one row per subject. The default is
  `--n_features 28`; set it to 2 for a two-feature table.
- Training and test numeric column names and their ordering must match.
  The current scaler processes the entire table after dropping `id` and
  `label`, before selecting the requested features. Do not add string
  metadata columns.
- Subjects in real training, validation, and test sets should be disjoint.
  Synthetic demos illustrate formats and execution only; their performance
  is not evidence of generalization. The generator reuses demonstration
  ID strings between training and test files.
- Tri-modal fusion requires matching IDs across gait, behavior, and both
  eye modalities. `--subject_order` must contain all test IDs without
  duplicates. The original code has alignment limitations for gait and
  the second eye modality; see `USAGE_NOTES.md`.

## Continuous-age scores

Inputs must include at least
`class_0,class_1,class_2,class_3,label`.
The score columns correspond to `8W,3M,12M,18M`. `label` is the true class
index, which the original function uses to select an age anchor.
`max_index` is not required for this step.

Scores are normalized by their row sums. Check for zero row sums, missing
values, and unexpected negative scores.
