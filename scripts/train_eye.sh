#!/usr/bin/env bash
# Run from anywhere. Paths in environment variables are relative to repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SPECIES="${SPECIES:-human}"
case "$SPECIES" in human|mouse) ;; *) echo "SPECIES must be human or mouse" >&2; exit 2 ;; esac
: "${DATA_ROOT:?Set DATA_ROOT to an ImageFolder root containing train/val/test}"
: "${HCF_CKPT:?Set HCF_CKPT to your pretrained checkpoint}"
MODEL="${MODEL:-vit_base_patch16}"
INPUT_SIZE="${INPUT_SIZE:-256}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NPROC="${NPROC:-1}"
NB_CLASSES="${NB_CLASSES:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${SPECIES}/eye/${MODEL}}"
TASK="${TASK:-eye-${INPUT_SIZE}/finetune-HCF-${EPOCHS}}"

"${PYTHON:-python}" -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
    "$ROOT/$SPECIES/train_eye.py" \
    --data_path "$DATA_ROOT" --finetune "$HCF_CKPT" \
    --model "$MODEL" --input_size "$INPUT_SIZE" --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" --world_size "$NPROC" --nb_classes "$NB_CLASSES" \
    --output_dir "$OUTPUT_ROOT" --log_dir "$OUTPUT_ROOT" --task "$TASK" \
    --blr "${BLR:-5e-4}" --layer_decay "${LAYER_DECAY:-0.65}" \
    --weight_decay 0.05 --drop_path "${DROP_PATH:-0.1}" \
    --reprob 0.25 --mixup 0.8 --cutmix 1.0 "$@"
