"""Train and evaluate classical ML classifiers on behavior-modality features.

The behavior modality uses open-field-test (OFT) pose-tracking summaries.
Each subject is represented by a single feature row (no aggregation), in
contrast to the gait modality.

Supported classifiers: baseline, AdaBoost, Bagging, DecisionTree, ExtraTree,
ExtraTrees, GaussianProcess, GradientBoosting, HistGradientBoosting,
KNeighbors, RandomForest, MLP, and a linear SGD model.

Example
-------
    python train_behavior.py \\
        --train_csv data_examples/behavior_train.csv \\
        --test_csv  data_examples/behavior_test.csv \\
        --n_features 28 \\
        --model_dir checkpoints/behavior/model-oft-time-3 \\
        --classifiers MLP LM

CSV columns: ``id``, ``feature1 ... featureN``, ``label``.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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
            loss="log_loss", n_estimators=100, random_state=random_state,
        ),
        "histgradientboosting": lambda: HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.2, random_state=random_state,
        ),
        "kneighbors": lambda: KNeighborsClassifier(
            n_neighbors=6, weights="distance", algorithm="auto",
            leaf_size=50, p=2,
        ),
        "randomforest": lambda: RandomForestClassifier(
            n_estimators=400, criterion="log_loss", random_state=0,
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


def preprocess(
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    imputer: SimpleImputer | None = None,
    scaler: MinMaxScaler | None = None,
) -> Tuple[pd.DataFrame, SimpleImputer, MinMaxScaler]:
    """Impute missing values and min-max scale the feature matrix."""
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


def train_and_save(
    classifier_name: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_dir: str,
) -> None:
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
    classifier = joblib.load(os.path.join(model_dir, f"model-{classifier_name}.pkl"))
    imputer = joblib.load(os.path.join(model_dir, f"imputer-{classifier_name}.pkl"))
    scaler = joblib.load(os.path.join(model_dir, f"scaler-{classifier_name}.pkl"))

    x_test = test_df.drop(columns=["id", "label"])
    y_true = test_df["label"].values

    x_scaled, _, _ = preprocess(x_test, feature_cols, imputer, scaler)
    y_pred = classifier.predict(x_scaled.values)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True, type=str)
    parser.add_argument("--test_csv", required=True, type=str)
    parser.add_argument("--n_features", type=int, default=28,
                        help="Number of feature columns (feature1 .. featureN).")
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--classifiers", nargs="+", default=["MLP"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    feature_cols = [f"feature{i + 1}" for i in range(args.n_features)]

    for name in args.classifiers:
        train_and_save(name, train_df, feature_cols, args.model_dir)
        evaluate_saved(name, test_df, feature_cols, args.model_dir)


if __name__ == "__main__":
    main()
