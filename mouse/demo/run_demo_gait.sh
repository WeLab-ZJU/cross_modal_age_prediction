#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
if [[ ! -f data_examples/gait_train.csv || ! -f data_examples/gait_test.csv ]]; then
    "$PYTHON" _generate_demo_data.py
fi
"$PYTHON" train_gait.py \
    --train_csv data_examples/gait_train.csv --test_csv data_examples/gait_test.csv \
    --feature_cols feature1 feature2 --model_dir outputs/demo_gait_model \
    --classifiers ExtraTrees RandomForest baseline "$@"
