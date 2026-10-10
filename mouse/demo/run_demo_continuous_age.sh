#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
if [[ ! -f data_examples/fused_features_example.csv ]]; then
    "$PYTHON" _generate_demo_data.py
fi
"$PYTHON" predict_continuous_age.py \
    --input_csv data_examples/fused_features_example.csv \
    --output_dir outputs/demo_continuous_age "$@"
