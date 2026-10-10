#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${OCT_CKPT:?Set OCT_CKPT}"
: "${OCTA_CKPT:?Set OCTA_CKPT}"
: "${OCT_TEST_ROOT:?Set OCT_TEST_ROOT}"
: "${OCTA_TEST_ROOT:?Set OCTA_TEST_ROOT}"
"${PYTHON:-python}" "$ROOT/run.py" human "${WORKFLOW:-fuse-eye-age}" \
    --oct_ckpt "$OCT_CKPT" --octa_ckpt "$OCTA_CKPT" \
    --oct_test_root "$OCT_TEST_ROOT" --octa_test_root "$OCTA_TEST_ROOT" \
    --oct_arch "${OCT_ARCH:-vit_base_patch16}" \
    --octa_arch "${OCTA_ARCH:-vit_base_patch16}" \
    --oct_img_size "${OCT_IMG_SIZE:-256}" --octa_img_size "${OCTA_IMG_SIZE:-256}" \
    --oct_device "${OCT_DEVICE:-cuda:0}" --octa_device "${OCTA_DEVICE:-cuda:0}" \
    --num_classes 4 --class_order "1-30" "30-60" "60-80" "80+" \
    --output_dir "${OUTPUT_ROOT:-outputs/human/fusion}" "$@"
