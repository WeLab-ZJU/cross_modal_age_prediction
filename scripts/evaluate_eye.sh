#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SPECIES="${SPECIES:-human}"
case "$SPECIES" in human|mouse) ;; *) echo "SPECIES must be human or mouse" >&2; exit 2 ;; esac
: "${DATA_ROOT:?Set DATA_ROOT to an ImageFolder root containing train/val/test}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint}"
"${PYTHON:-python}" "$ROOT/run.py" "$SPECIES" evaluate-eye \
    --data_path "$DATA_ROOT" --resume "$CHECKPOINT" \
    --model "${MODEL:-vit_base_patch16}" --input_size "${INPUT_SIZE:-256}" \
    --nb_classes "${NB_CLASSES:-4}" --batch_size "${BATCH_SIZE:-12}" \
    --device "${DEVICE:-cuda}" --num_workers "${NUM_WORKERS:-4}" \
    --output_dir "${OUTPUT_ROOT:-outputs/${SPECIES}/evaluation}" \
    --task "${TASK:-eye}" "$@"
