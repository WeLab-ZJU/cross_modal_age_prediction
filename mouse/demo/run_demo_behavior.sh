#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
if [[ ! -f data_examples/behavior_train.csv || ! -f data_examples/behavior_test.csv ]]; then
    "$PYTHON" _generate_demo_data.py
fi
"$PYTHON" train_behavior.py \
    --train_csv data_examples/behavior_train.csv --test_csv data_examples/behavior_test.csv \
    --n_features 28 --model_dir outputs/demo_behavior_model \
    --classifiers ExtraTrees MLP LM "$@"
