"""Generate small synthetic CSV files so the demos can run end-to-end.

The generated files mimic the shape of the real datasets but contain only
random values drawn from per-class Gaussians. Running this script twice will
produce identical outputs because the seeds are fixed.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd


def _make_gait_df(n_per_class: int, n_seq_per_subject: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    subject_counter = 0
    for class_idx in range(4):
        centre = np.array([class_idx * 1.5, 4 - class_idx])
        for _ in range(n_per_class):
            subject_id = f"S{subject_counter:03d}"
            subject_counter += 1
            for _ in range(n_seq_per_subject):
                feats = centre + rng.normal(0, 0.6, size=2)
                rows.append({
                    "id": subject_id,
                    "feature1": float(feats[0]),
                    "feature2": float(feats[1]),
                    "feature3": float(rng.normal()),
                    "feature4": float(rng.normal()),
                    "feature5": float(rng.normal()),
                    "feature6": float(rng.normal()),
                    "label": class_idx,
                })
    return pd.DataFrame(rows)


def _make_behavior_df(n_per_class: int, n_features: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    subject_counter = 0
    for class_idx in range(4):
        centre = rng.normal(class_idx * 0.8, 0.1, size=n_features)
        for _ in range(n_per_class):
            subject_id = f"B{subject_counter:03d}"
            subject_counter += 1
            feats = centre + rng.normal(0, 0.4, size=n_features)
            row = {"id": subject_id}
            row.update({f"feature{i + 1}": float(v) for i, v in enumerate(feats)})
            row["label"] = class_idx
            rows.append(row)
    return pd.DataFrame(rows)


def _make_fused_features_csv(rng: np.random.Generator) -> pd.DataFrame:
    """Emulate the output of fuse_modalities.py (class_0..3 + max_index + label)."""
    rows = []
    for class_idx in range(4):
        for _ in range(6):
            scores = rng.normal(0.25, 0.15, size=4)
            scores[class_idx] += 0.5 + rng.normal(0, 0.1)
            rows.append({
                "class_0": float(scores[0]),
                "class_1": float(scores[1]),
                "class_2": float(scores[2]),
                "class_3": float(scores[3]),
                "max_index": int(np.argmax(scores)),
                "label": class_idx,
            })
    return pd.DataFrame(rows)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng_train = np.random.default_rng(seed=0)
    rng_test = np.random.default_rng(seed=1)
    rng_fused = np.random.default_rng(seed=2)

    gait_train = _make_gait_df(n_per_class=12, n_seq_per_subject=4, rng=rng_train)
    gait_test = _make_gait_df(n_per_class=3, n_seq_per_subject=4, rng=rng_test)
    gait_train.to_csv(os.path.join(out_dir, "gait_train.csv"), index=False)
    gait_test.to_csv(os.path.join(out_dir, "gait_test.csv"), index=False)

    behavior_train = _make_behavior_df(n_per_class=12, n_features=28, rng=rng_train)
    behavior_test = _make_behavior_df(n_per_class=3, n_features=28, rng=rng_test)
    behavior_train.to_csv(os.path.join(out_dir, "behavior_train.csv"), index=False)
    behavior_test.to_csv(os.path.join(out_dir, "behavior_test.csv"), index=False)

    fused = _make_fused_features_csv(rng_fused)
    fused.to_csv(os.path.join(out_dir, "fused_features_example.csv"), index=False)

    print(f"Wrote demo CSVs to {out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(os.path.join(here, "data_examples"))
