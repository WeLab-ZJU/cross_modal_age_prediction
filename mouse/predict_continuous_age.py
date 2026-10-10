"""Map per-class probabilities to a continuous biological-age score.

This module turns the four-way class probabilities produced by
`fuse_modalities.py` (columns ``class_0 ... class_3``, one row per subject,
plus a ground-truth ``label``) into a scalar biological-age estimate, then
visualises the score distribution for every age cohort.

Score definition
----------------
For a subject whose true cohort is ``i`` with cohort age anchor ``A[i]``:

    row_sum        = sum_j p_j
    w_j            = p_j / row_sum                # normalised probabilities
    S_age          = A[i] + sum_{j != i} w_j * c_j

where ``c_j`` are fixed age-gap coefficients. Intuitively, the score starts
at the cohort anchor and is shifted by the probability mass that the fused
model assigns to *other* cohorts.

Example
-------
    python predict_continuous_age.py \\
        --input_csv  data_examples/fused_features_example.csv \\
        --output_dir outputs/continuous_age
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


# Four-way age cohorts (weeks converted to months for 8W): 8W, 3M, 12M, 18M
DEFAULT_FEATURE_COLS = ("class_0", "class_1", "class_2", "class_3")
DEFAULT_AGE_ANCHORS = (2.0, 3.0, 12.0, 18.0)                       # months
DEFAULT_COEFFICIENTS = (2.5 / 2.0, 7.5 / 2.0, 10.5 / 2.0, 18.0)    # gap weights
DEFAULT_CLASS_LABELS = ("2M", "3M", "12M", "18M")


def compute_age_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    age_anchors: Sequence[float] = DEFAULT_AGE_ANCHORS,
    coefficients: Sequence[float] = DEFAULT_COEFFICIENTS,
) -> pd.DataFrame:
    """Add a continuous ``S_age`` column to ``df`` and return it.

    Parameters
    ----------
    df:
        DataFrame containing ``feature_cols`` plus a ``label`` column whose
        integer values index into ``age_anchors`` and ``coefficients``.
    feature_cols:
        Probability-like columns (one per cohort). Rows are re-normalised to
        sum to 1 before scoring.
    age_anchors:
        Per-cohort age anchors (same order as ``feature_cols``).
    coefficients:
        Per-cohort age-gap coefficients applied to the *other* cohorts.
    """
    if len(feature_cols) != len(age_anchors) or len(feature_cols) != len(coefficients):
        raise ValueError(
            "feature_cols, age_anchors and coefficients must all have the same length."
        )

    features = list(feature_cols)
    row_sums = df[features].sum(axis=1)
    normalised = df[features].div(row_sums, axis=0)

    anchor_per_row = []
    masked_coeffs = []
    for true_class in df["label"]:
        anchor_per_row.append(age_anchors[true_class])
        masked_coeffs.append([
            0.0 if i == true_class else c
            for i, c in enumerate(coefficients)
        ])
    coeff_df = pd.DataFrame(masked_coeffs, columns=features, index=df.index)

    weighted = normalised * coeff_df
    df = df.copy()
    df["S_age"] = pd.Series(anchor_per_row, index=df.index) + weighted.sum(axis=1)
    return df


def plot_age_distribution(
    df: pd.DataFrame,
    output_path: str,
    class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
) -> None:
    """Plot per-cohort mean ``S_age`` with variance error bars."""
    grouped = df.groupby("label")["S_age"]
    means = grouped.mean()
    variances = grouped.var()

    n_classes = len(class_labels)
    palette = plt.get_cmap("Set2").colors[:n_classes]

    plt.figure(figsize=(10, 6))
    plt.bar(
        means.index, means, yerr=variances,
        capsize=4, color=palette[: len(means)], alpha=0.85,
        edgecolor="black", linewidth=1.2,
    )

    for i, (mu, var) in enumerate(zip(means, variances)):
        plt.text(
            means.index[i], mu + max(means) * 0.05,
            f"mu={mu:.2f}\nvar={var:.2f}",
            ha="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )

    plt.xlabel("Cohort label", fontsize=12, fontweight="bold")
    plt.ylabel("Mean S_age (months)", fontsize=12, fontweight="bold")
    plt.title("Continuous age score by cohort",
              fontsize=14, fontweight="bold")
    plt.xticks(means.index, [class_labels[i] for i in means.index])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(-1, n_classes)
    plt.ylim(0, float(max(means)) * 1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run(input_csv: str, output_dir: str) -> None:
    """End-to-end: compute scores, save CSV, and save figure."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    scored = compute_age_scores(df)

    out_csv = os.path.join(output_dir, "continuous_age_scores.csv")
    scored.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_png = os.path.join(output_dir, "continuous_age_scores.png")
    plot_age_distribution(scored, out_png)
    print(f"Wrote {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, type=str,
                        help="Fused-feature CSV with class_0..3 + label columns.")
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
