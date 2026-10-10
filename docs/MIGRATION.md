# Migration Record and Known Limitations

Consolidation and English-edition date: October 10, 2026.

## Consolidation principles

- The original `codes/` and `codes_mouse/` directories were not modified.
  Consolidation took place in a new output directory.
- All original Python files were retained. `source_manifest.json` records
  file origins and original and packaged SHA-256 hashes, using relative
  paths rather than personal machine paths.
- Training, model, and `util/` code originally matched between the two
  sources. Separate copies were retained to reduce refactoring risk.
- `codes/A_train_eye.py` originally matched `train_eye.py`. It remains
  available as a historical duplicate entry point.
- `codes/fuse_modalities.py` and `_generate_demo_data.py` originally matched
  the mouse versions. They remain under `human/` for completeness but are
  not registered as human workflows in the unified launcher.
- The missing `mouse/engine_finetune.py` was supplied from the original
  human source directory.
- Original shell scripts and READMEs were moved to `legacy/<species>/`.
  Scripts are stored as `.sh.txt` to distinguish them from current
  launchers. Old machine paths were replaced with angle-bracket
  placeholders. External-source, performance, environment, and licensing
  claims in historical READMEs were **not verified during consolidation**.
  Treat those files as historical records; use the root README for
  current instructions.
- `.DS_Store`, `__pycache__`, `.pyc`, and old `.gitignore` files were
  excluded.
- The root README, data conventions, dependency lists, unified launcher,
  configurable shell scripts, and standard-library checks were added.

## Limited Python changes made during consolidation

1. `human/train_eye.py`, `human/A_train_eye.py`, and `mouse/train_eye.py`:
   enabled argparse `--help` without changing training defaults.
2. Both `_generate_demo_data.py` copies: changed the output directory from
   the script directory to its `data_examples/` subdirectory, matching the
   demo scripts.
3. `human/visualize_all_best_models.py`: replaced old server defaults with
   `outputs/human/...` and `data/human`.
4. `mouse/engine_finetune.py`: supplied the missing dependency, identical
   to the original human engine before translation.

No loss function, model architecture, checkpoint format, class reordering,
fusion algorithm, default training hyperparameter, or metric formula was
changed.

## English edition

- `cross_modal_age_prediction_en/` is an independent sibling copy of the
  consolidated repository; the existing version remains unchanged.
- The root README, data conventions, migration notes, validation report,
  and legacy overview were translated into English.
- Chinese comments and docstrings in the fusion scripts and training
  engines, as well as comments in historical FP scripts, were translated.
  One Chinese `ValueError` message was translated without changing its
  exception type or condition.
- Filenames, command-line options, identifiers, formulas, defaults, and
  directory relationships were preserved.
- Original-source hashes in `source_manifest.json` are retained.
  Packaged hashes reflect this English edition, and translated source
  entries are marked in their action descriptions.
- Historical scientific claims remain historical and unverified;
  translation does not endorse or correct their scientific interpretation.

## Original-code issues identified but not changed

The following are source-review findings. They do not imply that each
issue has been reproduced in the original experimental environment.

### Eye training and dependencies

- No original environment lock file was provided. The model subclasses
  timm's ViT; verify compatibility between `forward_features` and the
  inherited `forward` / classification head in the actual installed
  version. In particular, global pooling returns `[B,1,C]`, while the
  classification loss generally expects `[B,K]`. Check whether the
  original environment handles this shape. No tensor-shape change was
  made during consolidation or translation.
- Some historical Base/16 templates use size 196, and Huge/14 templates
  use size 256. These are not integer multiples of their patch sizes.
  Verify patch embedding and checkpoint compatibility before reuse. The
  new template defaults to Base/16 with size 256.
- Training defaults to `global_pool=True`. Both pipelines use CUDA AMP,
  and `train_one_epoch()` calls `torch.cuda.synchronize()` unconditionally.
  CPU/MPS training support has not been added.
- `evaluate()` returns `(F1 + ROC_AUC + Kappa)/3` as the selection score,
  although the training variable is named `max_auc`. Only the
  documentation was corrected.
- If validation lacks a class and AUC becomes NaN, or all scores fail to
  exceed the initial value of 0, no best checkpoint may be saved.
  The training script nevertheless attempts to load
  `checkpoint-best.pth` at the end.
- Distributed evaluation synchronizes metric counters without gathering
  complete predictions across ranks before computing every metric.
  Multiple ranks may also write the same CSV or confusion-matrix file.
  The new launcher defaults to one GPU. Review multi-GPU evaluation
  separately, especially `--dist_eval`.
- `--eval` still constructs train, validation, and test datasets. A test
  directory alone is insufficient.
- Best checkpoints do not include the full optimizer state. Resuming
  from one is not equivalent to fully restoring training.
- Verify checkpoint deserialization behavior in the actual PyTorch
  version. Do not unconditionally trust unknown serialized files merely
  to resolve compatibility issues.

### Historical shell templates

- Some Huge templates expose only two GPUs but launch four processes.
- A single-modality FP evaluation template passes `--class_order`, which
  the training entry point does not accept.
- The old `vis.sh` references a missing `vis_vit_cam.py`; the old
  `test_eye_huge.sh` is empty.
- Several variables are assigned repeatedly, with later assignments
  overriding earlier ones. These settings remain in the historical
  records. Use the new `scripts/` directory for configurable launchers.

### Mouse fusion and data semantics

- Devices are selected automatically: CPU without CUDA, one shared GPU
  when only one is available, or `cuda:0` and `cuda:1` with multiple GPUs.
  No device CLI was added.
- The second eye modality is reordered using the first modality's ID
  list. Source files must align strictly by subject; equal file counts
  alone do not establish correct correspondence.
- Aggregated gait features retain the order of each subject's first
  appearance in the CSV. They are not explicitly reordered using
  `--subject_order`. Ensure that these orders agree before invocation.
  The original label-only check cannot detect mismatches within the same
  age group.
- Eye files are paired by sorting and `zip`, which can truncate unequal
  lists. Verify filenames, IDs, labels, and ordering for formal evaluation.
- Outputs called “Eye-A-Single” and “Eye-B-Single” originate from
  `head(latent1 + latent2)` in the original code. They are not isolated
  single-modality inference without the other modality.
- Synthetic behavior and gait IDs differ. No fully synthetic, real-data-free
  tri-modal demo is advertised. The historical fusion template remains
  under `legacy/mouse/demo/`; see the root README for the current command.
- Continuous-age scoring requires the true cohort `label`; it should not
  be described as entirely independent age prediction.

## Publication status

Consolidation covers packaging, path cleanup, and static checks.
Translation does not add training, performance comparisons, or real-data
pairing validation. Confirm the limitations above in the original
environment before reproducing scientific results.

No LICENSE was added. Confirm author permissions, third-party origins,
and compatible licensing terms before publication.
