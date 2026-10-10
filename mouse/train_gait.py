"""Train and evaluate classical ML classifiers on gait-modality features.

Each subject contributes multiple gait sequences; per-sequence class probabilities
are aggregated per subject (sum-then-argmax) before scoring, which matches how
the model is consumed downstream by `fuse_modalities.py`.

Supported classifiers: baseline (majority-class), AdaBoost, Bagging,
DecisionTree, ExtraTree, ExtraTrees, GaussianProcess, GradientBoosting,
KNeighbors, RandomForest, MLP, and a linear SGD model.

Example
-------
    python train_gait.py \\
        --train_csv data_examples/gait_train.csv \\
        --test_csv  data_examples/gait_test.csv \\
        --feature_cols feature1 feature2 \\
        --model_dir  checkpoints/gait/model-BT6-3 \\
        --classifiers ExtraTrees RandomForest

The training CSV must contain an ``id`` column (subject identifier), the
feature columns, and a ``label`` column with integer class indices.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


CLASS_NAMES = ["8W", "3M", "12M", "18M"]
CLASS_INDICES = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------
def build_classifier(name: str, random_state: int = 521):
    """Return an unfitted sklearn classifier keyed by ``name``."""
    name = name.lower()
    builders = {
        "baseline": lambda: DummyClassifier(),
        "adaboost": lambda: AdaBoostClassifier(
            n_estimators=400, learning_rate=0.25,
            algorithm="SAMME", random_state=random_state,
        ),
        "bagging": lambda: BaggingClassifier(
            n_estimators=400, random_state=random_state,
        ),
        "decisiontree": lambda: DecisionTreeClassifier(
            criterion="gini", splitter="best", random_state=random_state,
        ),
        "extratree": lambda: ExtraTreeClassifier(
            criterion="gini", splitter="best", random_state=random_state,
        ),
        "extratrees": lambda: ExtraTreesClassifier(
            n_estimators=100, random_state=random_state,
        ),
        "gaussianprocess": lambda: GaussianProcessClassifier(),
        "gradientboosting": lambda: GradientBoostingClassifier(
            n_estimators=100, random_state=random_state,
        ),
        "kneighbors": lambda: KNeighborsClassifier(
            n_neighbors=6, weights="distance", algorithm="auto",
            leaf_size=50, p=2,
        ),
        "randomforest": lambda: RandomForestClassifier(
            n_estimators=400, random_state=0,
        ),
        "mlp": lambda: MLPClassifier(
            solver="sgd", activation="relu", alpha=1e-5, batch_size=4,
            hidden_layer_sizes=(64, 256, 64), random_state=1,
            max_iter=2000, verbose=False,
            learning_rate_init=0.02, learning_rate="invscaling",
        ),
        "lm": lambda: SGDClassifier(
            loss="log_loss", max_iter=1000, random_state=42,
        ),
    }
    if name not in builders:
        raise ValueError(
            f"Unknown classifier '{name}'. Choose from: {sorted(builders)}"
        )
    return builders[name]()


# ---------------------------------------------------------------------------
# Preprocessing & subject-level aggregation
# ---------------------------------------------------------------------------
def preprocess(
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    imputer: SimpleImputer | None = None,
    scaler: MinMaxScaler | None = None,
) -> Tuple[pd.DataFrame, SimpleImputer, MinMaxScaler]:
    """Impute missing values and min-max scale the feature matrix.

    When ``imputer`` / ``scaler`` are None they are fitted on ``features``.
    """
    features = features.copy()
    if imputer is None:
        imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        imputer.fit(features[list(feature_cols)])
    features[list(feature_cols)] = imputer.transform(features[list(feature_cols)])

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(features)
    scaled = pd.DataFrame(
        scaler.transform(features), columns=features.columns
    )[list(feature_cols)]
    return scaled, imputer, scaler


def aggregate_by_subject(
    subject_ids: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum per-sequence probabilities within each subject and take argmax.

    Returns the aggregated prediction, the summed probability matrix, and the
    (deduplicated) subject-level labels.
    """
    if len(subject_ids) != probabilities.shape[0]:
        raise ValueError("subject_ids and probabilities must have the same length.")
    if len(subject_ids) != labels.size:
        raise ValueError("subject_ids and labels must have the same length.")

    prob_df = pd.DataFrame(probabilities, index=subject_ids)
    summed = prob_df.groupby(level=0, sort=False).sum()
    y_pred = summed.values.argmax(axis=1)

    label_series = pd.Series(labels, index=subject_ids)
    y_true = label_series.groupby(level=0, sort=False).first().to_numpy()

    return y_pred, summed.values, y_true


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_and_save(
    classifier_name: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_dir: str,
) -> None:
    """Fit a classifier on the training set and persist artefacts to disk."""
    os.makedirs(model_dir, exist_ok=True)

    y_train = train_df["label"].values
    x_train = train_df.drop(columns=["id", "label"])
    x_scaled, imputer, scaler = preprocess(x_train, feature_cols)

    classifier = build_classifier(classifier_name)
    classifier.fit(x_scaled.values, y_train)

    joblib.dump(classifier, os.path.join(model_dir, f"model-{classifier_name}.pkl"))
    joblib.dump(imputer, os.path.join(model_dir, f"imputer-{classifier_name}.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler-{classifier_name}.pkl"))


def evaluate_saved(
    classifier_name: str,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_dir: str,
) -> Dict[str, float]:
    """Load a persisted classifier and compute per-subject metrics."""
    classifier = joblib.load(os.path.join(model_dir, f"model-{classifier_name}.pkl"))
    imputer = joblib.load(os.path.join(model_dir, f"imputer-{classifier_name}.pkl"))
    scaler = joblib.load(os.path.join(model_dir, f"scaler-{classifier_name}.pkl"))

    subject_ids = test_df["id"].values
    x_test = test_df.drop(columns=["id", "label"])
    y_test_raw = test_df["label"].values

    x_scaled, _, _ = preprocess(x_test, feature_cols, imputer, scaler)
    proba = classifier.predict_proba(x_scaled.values)

    y_pred, _, y_true = aggregate_by_subject(subject_ids, proba, y_test_raw)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }

    print(f"\n=== {classifier_name} ===")
    print(f"pred : {y_pred.tolist()}")
    print(f"label: {y_true.tolist()}")
    for key, value in metrics.items():
        print(f"  {key:17s}: {value:.4f}")

    print("  Per-class metrics:")
    for idx, class_idx in enumerate(CLASS_INDICES):
        binary_true = (y_true == class_idx)
        binary_pred = (y_pred == class_idx)
        cp = precision_score(binary_true, binary_pred, zero_division=0)
        cr = recall_score(binary_true, binary_pred, zero_division=0)
        cf = f1_score(binary_true, binary_pred, zero_division=0)
        print(f"    {CLASS_NAMES[idx]:>4s}  F1={cf:.4f}  P={cp:.4f}  R={cr:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--feature_cols", nargs="+", required=True,
                        help="Names of feature columns, e.g. feature1 feature2.")
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory where model/imputer/scaler are saved.")
    parser.add_argument("--classifiers", nargs="+", default=["ExtraTrees"],
                        help="List of classifier names to train and evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    for name in args.classifiers:
        train_and_save(name, train_df, args.feature_cols, args.model_dir)
        evaluate_saved(name, test_df, args.feature_cols, args.model_dir)


if __name__ == "__main__":
    main()
