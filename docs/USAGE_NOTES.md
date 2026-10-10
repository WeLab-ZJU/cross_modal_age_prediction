# Usage Notes

## Environment and eye models

- Dependency versions are not pinned. The ViT implementation subclasses
  timm; check compatibility between `forward_features`, the inherited
  `forward`, and the classification head in your environment. Global
  pooling returns `[B,1,C]`; the classification loss expects `[B,K]`.
  Verify the output shape with the installed timm version.
- Match input dimensions to the patch size and checkpoint. Some historical
  Base/16 configurations use size 196 and Huge/14 configurations use size
  256, neither divisible by the corresponding patch size. The current
  launcher defaults to Base/16 with size 256.
- Eye training uses CUDA AMP and an unconditional
  `torch.cuda.synchronize()` call. CPU/MPS training is not supported.
- The default pooling setting is `global_pool=True`.
- Load only trusted checkpoints and joblib files. Check deserialization
  compatibility with your PyTorch version.

## Training and evaluation

- `--eval` still constructs `train/`, `val/`, and `test/`; all are required.
  Use `--resume` for evaluation weights.
- The best-checkpoint selection score is `(F1 + ROC_AUC + Kappa)/3`,
  although the training variable is named `max_auc`.
- Validation needs adequate class coverage. Missing classes can make AUC
  NaN. If no score exceeds the initial value of 0, a best checkpoint may
  not be saved, while the final evaluation still attempts to load it.
- Best checkpoints do not contain the full optimizer state. Loading one
  does not fully restore an interrupted training run.
- Distributed evaluation synchronizes counters but does not gather all
  predictions across ranks before computing every metric. Multiple ranks
  can also write the same CSV or confusion-matrix file. The launcher
  defaults to one GPU; review this behavior when using multiple ranks,
  especially with `--dist_eval`.

## Mouse fusion and age scores

- Device selection is automatic: CPU without CUDA, a shared GPU with one
  GPU, or `cuda:0` and `cuda:1` with multiple GPUs.
- Both eye modalities must be aligned by subject. The second modality is
  reordered using the first modality's ID list. Equal file counts do not
  establish correspondence.
- Eye files are sorted and paired using `zip`, which truncates unequal
  lists. Check both file lists, subject IDs, labels, and ordering.
- Aggregated gait features retain each subject's first-occurrence order
  in the CSV; they are not explicitly reordered by `--subject_order`.
  Ensure these orders match. The label-only check cannot detect swapped
  subjects within an age group.
- Outputs named “Eye-A-Single” and “Eye-B-Single” use
  `head(latent1 + latent2)` and are not isolated single-modality inference.
- Synthetic gait and behavior demos use different IDs and cannot be
  directly combined into tri-modal fusion.
- Continuous-age scoring uses the true cohort `label` to choose an age
  anchor. It is not independent age regression without cohort information.

## Historical configurations

Use `scripts/` and `mouse/demo/` for runnable launchers. Files in `legacy/`
are configuration references stored as `.sh.txt`.

- Some Huge references expose two GPUs but request four processes.
- The historical single-modality FP test passes `--class_order`, which
  `train_eye.py` does not accept.
- The historical `vis.sh` references an unavailable `vis_vit_cam.py`;
  `test_eye_huge.sh` is empty. Use `human/visualize_all_best_models.py` for
  the provided visualization workflow.
- Repeated variable assignments use their last assigned value.
