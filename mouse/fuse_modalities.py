"""Fuse eye, gait and behavior modalities into a single per-subject feature.

Pipeline
--------
1. Load two fine-tuned ViT backbones (one per eye sub-modality, e.g. fundus
   and OCTA) and run them on the paired test images to obtain per-class
   logit vectors. Logits are cross-summed across the two views before the
   classification head (``run_two_image_cross``).
2. Load the saved behaviour classifier and call ``predict_proba`` to obtain
   a per-subject probability vector.
3. Load the saved gait classifier, obtain per-sequence probabilities, and
   aggregate to subject level by summing.
4. L2-normalise every modality vector and fuse them (optionally with fixed
   per-modality weights). Seven fusion variants are evaluated and their
   metrics are written to an Excel report; every variant also saves a CSV
   of per-subject fused features for downstream scripts such as
   ``predict_continuous_age.py``.

The script is intentionally configurable via command-line arguments so that
users can point at their own checkpoints and CSVs.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

import models_vit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLASS_ORDER_EYE = ("12M", "18M", "3M", "8W")           # directory order
CLASS_ORDER_REPORT = ("8W", "3M", "12M", "18M")        # reporting order

METHOD_LABELS: Dict[str, str] = {
    "single_feature_predict_eye_A": "Eye-A-Single",
    "single_feature_predict_eye_B": "Eye-B-Single",
    "cross_feature_predict_eye": "Two-Eye",
    "cross_feature_predict_eye_gait": "Two-Eye+Gait",
    "cross_feature_predict_eye_behavior": "Two-Eye+Behavior",
    "cross_feature_predict_gait_behavior": "Gait+Behavior",
    "cross_feature_predict_all": "Tri-Modal",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def reorder_list_by_ids(items: list, old_ids: Sequence[str],
                        new_ids: Sequence[str]) -> list:
    """Reorder ``items`` so that their positions follow ``new_ids``.

    ``old_ids`` and ``new_ids`` must contain exactly the same identifiers.
    """
    if len(items) != len(old_ids) or len(old_ids) != len(new_ids):
        raise ValueError("All inputs must have equal length.")
    if set(old_ids) != set(new_ids):
        raise ValueError("new_ids must be a permutation of old_ids.")
    old_to_idx = {oid: idx for idx, oid in enumerate(old_ids)}
    return [items[old_to_idx[nid]] for nid in new_ids]


def reorder_df_by_ids(df: pd.DataFrame, old_ids: Sequence[str],
                      new_ids: Sequence[str]) -> pd.DataFrame:
    """Reorder the rows of ``df`` to match ``new_ids``."""
    if len(df) != len(old_ids) or len(old_ids) != len(new_ids):
        raise ValueError("All inputs must have equal length.")
    old_to_idx = {oid: idx for idx, oid in enumerate(old_ids)}
    order = [old_to_idx[nid] for nid in new_ids]
    return df.iloc[order].reset_index(drop=True)


def aggregate_by_subject(subject_ids: np.ndarray,
                         probabilities: np.ndarray,
                         labels: np.ndarray,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum per-sequence probabilities within each subject, take argmax."""
    if len(subject_ids) != probabilities.shape[0]:
        raise ValueError("subject_ids and probabilities length mismatch.")
    if len(subject_ids) != labels.size:
        raise ValueError("subject_ids and labels length mismatch.")
    prob_df = pd.DataFrame(probabilities, index=subject_ids)
    summed = prob_df.groupby(level=0, sort=False).sum()
    y_pred = summed.values.argmax(axis=1)
    y_true = pd.Series(labels, index=subject_ids) \
        .groupby(level=0, sort=False).first().to_numpy()
    return y_pred, summed.values, y_true


def dedup_by_subject_key(names: Sequence[str], feats: list,
                         strip_tokens: Sequence[str] = (
                             "8W", "12M", "3M", "19M", "18M", "14M")
                         ) -> Tuple[List[str], list]:
    """Keep only the first occurrence per subject key.

    Each ``name`` is expected to follow a pattern where the subject key can
    be recovered as ``name.split('-')[-1].split('_')[0]``. Tokens in
    ``strip_tokens`` are removed from the key.
    """
    seen, keys, out_feats = set(), [], []
    for n, f in zip(names, feats):
        key = n.split("-")[-1].split("_")[0]
        for token in strip_tokens:
            key = key.replace(token, "")
        if key not in seen:
            seen.add(key)
            keys.append(key)
            out_feats.append(f)
    return keys, out_feats


def row_normalize_l2(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("Input must be a 2-D matrix.")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / (norms + eps)


# ---------------------------------------------------------------------------
# Tabular-modality preprocessing / prediction
# ---------------------------------------------------------------------------
def preprocess_tabular(df: pd.DataFrame,
                       feature_cols: Sequence[str],
                       imputer: SimpleImputer | None = None,
                       scaler: MinMaxScaler | None = None,
                       ) -> Tuple[pd.DataFrame, SimpleImputer, MinMaxScaler]:
    df = df.copy()
    if imputer is None:
        imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        imputer.fit(df[list(feature_cols)])
    df[list(feature_cols)] = imputer.transform(df[list(feature_cols)])

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)
    scaled = pd.DataFrame(scaler.transform(df),
                          columns=df.columns)[list(feature_cols)]
    return scaled, imputer, scaler


def predict_tabular(df: pd.DataFrame,
                    feature_cols: Sequence[str],
                    model_path: str,
                    imputer_path: str,
                    scaler_path: str,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    classifier = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    x, _, _ = preprocess_tabular(df, feature_cols, imputer, scaler)
    x_values = x.values
    return classifier.predict(x_values), classifier.predict_proba(x_values)


# ---------------------------------------------------------------------------
# Eye-modality inference
# ---------------------------------------------------------------------------
def build_eye_model(checkpoint_path: str, arch: str,
                    img_size: int, num_classes: int = 4) -> torch.nn.Module:
    model = models_vit.__dict__[arch](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=0.2,
        global_pool=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return model


def load_and_normalize(path: str, img_size: int) -> np.ndarray:
    img = Image.open(path).resize((img_size, img_size))
    img = np.array(img) / 255.0
    assert img.shape == (img_size, img_size, 3)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def forward_pair_cross(imgs1: List[np.ndarray], model1: torch.nn.Module,
                       imgs2: List[np.ndarray], model2: torch.nn.Module,
                       device1: torch.device, device2: torch.device,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a cross-modal forward pass over two paired image batches."""
    def _to_batch(imgs: List[np.ndarray], device: torch.device) -> torch.Tensor:
        tensors = [
            torch.einsum("nhwc->nchw", torch.tensor(img).unsqueeze(dim=0))
            for img in imgs
        ]
        return torch.cat(tensors).to(device, non_blocking=True).float()

    x1 = _to_batch(imgs1, device1)
    x2 = _to_batch(imgs2, device2)

    latent1 = model1.forward_features(x1)
    latent2 = model2.forward_features(x2)

    res1 = torch.sum(model1.head(latent1 + latent2.to(device1)), dim=0).to("cpu")
    res2 = torch.sum(model2.head(latent1.to(device2) + latent2.to(device2)),
                     dim=0).to("cpu")
    return res1 + res2, res1, res2


def generate_augmented_names(file_name: str) -> List[str]:
    base = file_name.split("-", 1)[1]
    return [f"{prefix}-{base}"
            for prefix in ("elastic", "flip1", "flip2", "origin", "zoom")]


def collect_eye_features(model1: torch.nn.Module, model2: torch.nn.Module,
                         data_path1: str, data_path2: str, label: str,
                         label_index: Sequence[str],
                         img_list1: List[str], img_list2: List[str],
                         img_size: int, is_augment: bool,
                         device1: torch.device, device2: torch.device,
                         ) -> Tuple[float, List[str], List[str],
                                    list, list, list]:
    """Run paired eye-modality inference for every subject in a cohort."""
    name1_list, name2_list = [], []
    feature_list, feature1_list, feature2_list = [], [], []
    acc = 0.0

    for i, j in zip(img_list1, img_list2):
        def load_views(fname: str, base_dir: str) -> List[np.ndarray]:
            if is_augment:
                views = [load_and_normalize(os.path.join(base_dir, n), img_size)
                         for n in generate_augmented_names(fname)]
                other = fname.replace("_L", "_R") if "_L" in fname else \
                        fname.replace("_R", "_L") if "_R" in fname else None
                if other is not None:
                    views.extend([
                        load_and_normalize(os.path.join(base_dir, n), img_size)
                        for n in generate_augmented_names(other)
                    ])
                return views
            views = [load_and_normalize(os.path.join(base_dir, fname), img_size)]
            other = fname.replace("_L", "_R") if "_L" in fname else \
                    fname.replace("_R", "_L") if "_R" in fname else None
            if other is not None:
                views.append(load_and_normalize(os.path.join(base_dir, other),
                                                img_size))
            return views

        imgs1 = load_views(i, data_path1)
        imgs2 = load_views(j, data_path2)

        res, res1, res2 = forward_pair_cross(
            imgs1, model1, imgs2, model2, device1, device2
        )

        if label_index[int(torch.argmax(res))] == label:
            acc += 1

        name1_list.append(i)
        name2_list.append(j)
        feature_list.append(res.detach().cpu().numpy())
        feature1_list.append(res1.detach().cpu().numpy())
        feature2_list.append(res2.detach().cpu().numpy())

    return acc, name1_list, name2_list, feature_list, feature1_list, feature2_list


# ---------------------------------------------------------------------------
# Fusion variants
# ---------------------------------------------------------------------------
DEFAULT_EYE_WEIGHT = 0.85
DEFAULT_TAB_WEIGHT = 0.69


def fuse_two_eye(f1: np.ndarray, f2: np.ndarray, weighted: bool) -> np.ndarray:
    f1, f2 = row_normalize_l2(f1), row_normalize_l2(f2)
    if weighted:
        return f1 * DEFAULT_EYE_WEIGHT + f2 * DEFAULT_EYE_WEIGHT
    return f1 + f2


def fuse_single(f1: np.ndarray) -> np.ndarray:
    return row_normalize_l2(f1)


def fuse_eye_gait(f1: np.ndarray, f2: np.ndarray, fg: np.ndarray,
                  weighted: bool) -> np.ndarray:
    f1, f2, fg = row_normalize_l2(f1), row_normalize_l2(f2), row_normalize_l2(fg)
    if weighted:
        return (f1 * DEFAULT_EYE_WEIGHT + f2 * DEFAULT_EYE_WEIGHT
                + fg * DEFAULT_TAB_WEIGHT)
    return f1 + f2 + fg


def fuse_eye_behavior(f1: np.ndarray, f2: np.ndarray, fb: np.ndarray,
                      weighted: bool) -> np.ndarray:
    f1, f2, fb = row_normalize_l2(f1), row_normalize_l2(f2), row_normalize_l2(fb)
    if weighted:
        return (f1 * DEFAULT_EYE_WEIGHT + f2 * DEFAULT_EYE_WEIGHT
                + fb * DEFAULT_TAB_WEIGHT)
    return f1 + f2 + fb


def fuse_gait_behavior(fg: np.ndarray, fb: np.ndarray,
                       weighted: bool) -> np.ndarray:
    fg, fb = row_normalize_l2(fg), row_normalize_l2(fb)
    if weighted:
        return fg * DEFAULT_TAB_WEIGHT + fb * DEFAULT_TAB_WEIGHT
    return fg + fb


def fuse_all(f1: np.ndarray, f2: np.ndarray,
             fg: np.ndarray, fb: np.ndarray,
             weighted: bool) -> np.ndarray:
    f1, f2 = row_normalize_l2(f1), row_normalize_l2(f2)
    fg, fb = row_normalize_l2(fg), row_normalize_l2(fb)
    if weighted:
        return (f1 * DEFAULT_EYE_WEIGHT + f2 * DEFAULT_EYE_WEIGHT
                + fg * DEFAULT_TAB_WEIGHT + fb * DEFAULT_TAB_WEIGHT)
    return f1 + f2 + fg + fb


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------
def evaluate_fusion(method_key: str, fused: np.ndarray, y_true: np.ndarray,
                    feature_dir: str, model_type: str,
                    img_size: int, epochs: int) -> Dict[str, float]:
    """Compute overall + per-class metrics and save per-subject features."""
    y_pred = fused.argmax(axis=1)

    # Persist fused features for downstream scripts
    os.makedirs(feature_dir, exist_ok=True)
    feat_df = pd.DataFrame(
        fused, columns=[f"class_{i}" for i in range(fused.shape[1])]
    )
    feat_df["max_index"] = y_pred
    feat_df["label"] = y_true
    out_csv = os.path.join(
        feature_dir,
        f"{method_key}_features_{model_type}_{img_size}_{epochs}.csv",
    )
    feat_df.to_csv(out_csv, index=False)

    metrics: Dict[str, float] = {
        "Methods": METHOD_LABELS[method_key],
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }

    # Per-class metrics in reporting order (8W, 3M, 12M, 18M)
    per_class = []
    per_auc = []
    for i in range(fused.shape[1]):
        binary_true = (y_true == i)
        binary_pred = (y_pred == i)
        per_class.append((
            f1_score(binary_true, binary_pred, zero_division=0),
            precision_score(binary_true, binary_pred, zero_division=0),
            recall_score(binary_true, binary_pred, zero_division=0),
        ))
        try:
            per_auc.append(roc_auc_score(binary_true, fused[:, i]))
        except ValueError:
            per_auc.append(0.5)

    # Original class order is ('12M', '18M', '3M', '8W') -> reorder to report
    reorder = [3, 2, 0, 1]  # 8W, 3M, 12M, 18M
    for report_i, src_i in enumerate(reorder):
        prefix = CLASS_ORDER_REPORT[report_i]
        f1c, prec, rec = per_class[src_i]
        metrics[f"{prefix}_F1"] = f1c
        metrics[f"{prefix}_Precision"] = prec
        metrics[f"{prefix}_Recall"] = rec
        metrics[f"{prefix}_AUC"] = per_auc[src_i]

    metrics["Total_AUC"] = float(np.mean(per_auc))
    return metrics


def save_report(all_results: List[Dict[str, float]], output_xlsx: str) -> None:
    """Write a formatted Excel summary of every fusion method."""
    column_order = [
        "Methods",
        "8W_F1", "8W_Precision", "8W_Recall", "8W_AUC",
        "3M_F1", "3M_Precision", "3M_Recall", "3M_AUC",
        "12M_F1", "12M_Precision", "12M_Recall", "12M_AUC",
        "18M_F1", "18M_Precision", "18M_Recall", "18M_AUC",
        "Accuracy", "F1", "Precision", "Recall", "Total_AUC", "R2", "MSE",
    ]
    df = pd.DataFrame(all_results)[column_order]

    for col in column_order[1:]:
        if col in ("R2", "MSE"):
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
        else:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%")

    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df.to_excel(output_xlsx, index=False)

    wb = load_workbook(output_xlsx)
    ws = wb.active
    for col in range(1, ws.max_column + 1):
        ws.cell(row=1, column=col).font = Font(bold=True)
        ws.cell(row=1, column=col).alignment = Alignment(
            horizontal="center", vertical="center"
        )
    header_fill = PatternFill(start_color="D3D3D3",
                              end_color="D3D3D3", fill_type="solid")
    for start, end, title in (
        ("B1", "E1", "8W"),
        ("F1", "I1", "3M"),
        ("J1", "M1", "12M"),
        ("N1", "Q1", "18M"),
    ):
        ws.merge_cells(f"{start}:{end}")
        ws[start].value = title
        ws[start].alignment = Alignment(horizontal="center", vertical="center")
        ws[start].font = Font(bold=True)
        ws[start].fill = header_fill
    wb.save(output_xlsx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Eye modality
    parser.add_argument("--eye_arch", default="vit_base_patch16",
                        choices=["vit_base_patch16", "vit_large_patch16",
                                 "vit_huge_patch14"])
    parser.add_argument("--eye_ckpt_a", required=True, type=str,
                        help="Checkpoint of the first eye sub-modality.")
    parser.add_argument("--eye_ckpt_b", required=True, type=str,
                        help="Checkpoint of the second eye sub-modality.")
    parser.add_argument("--eye_test_root_a", required=True, type=str,
                        help="ImageFolder root of the first eye test set.")
    parser.add_argument("--eye_test_root_b", required=True, type=str,
                        help="ImageFolder root of the second eye test set.")
    parser.add_argument("--eye_img_size", type=int, default=256)
    parser.add_argument("--eye_epochs", type=int, default=700,
                        help="Used only as a tag in the report filename.")
    parser.add_argument("--eye_is_augment", action="store_true",
                        help="Indicate that augmented 5-view filenames are used.")

    # Behavior modality
    parser.add_argument("--behavior_test_csv", required=True, type=str)
    parser.add_argument("--behavior_model_dir", required=True, type=str)
    parser.add_argument("--behavior_classifier", default="ExtraTrees")
    parser.add_argument("--behavior_n_features", type=int, default=28)

    # Gait modality
    parser.add_argument("--gait_test_csv", required=True, type=str)
    parser.add_argument("--gait_model_dir", required=True, type=str)
    parser.add_argument("--gait_classifier", default="ExtraTrees")
    parser.add_argument("--gait_feature_cols", nargs="+",
                        default=["feature1", "feature2"])

    # Subject ordering: eye results are aggregated in this order before fusion
    parser.add_argument("--subject_order", nargs="+", required=True,
                        help="Subject IDs in the order expected by the tabular "
                             "classifiers.")

    parser.add_argument("--output_feature_dir",
                        default="outputs/fused_features")
    parser.add_argument("--output_report_xlsx",
                        default="outputs/reports/fusion_report.xlsx")
    parser.add_argument("--use_fixed_weights", action="store_true",
                        help="Apply the per-modality weights from the paper.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1
                           else device1)

    # --- Eye modality inference -------------------------------------------
    model_a = build_eye_model(args.eye_ckpt_a, args.eye_arch, args.eye_img_size)
    model_b = build_eye_model(args.eye_ckpt_b, args.eye_arch, args.eye_img_size)
    model_a.to(device1).eval()
    model_b.to(device2).eval()

    name_a_all, feat_a_all = [], []
    name_b_all, feat_b_all = [], []
    label_all: List[int] = []

    for cohort_idx, cohort in enumerate(CLASS_ORDER_EYE):
        path_a = os.path.join(args.eye_test_root_a, cohort)
        path_b = os.path.join(args.eye_test_root_b, cohort)
        list_a = sorted(os.listdir(path_a))
        list_b = sorted(os.listdir(path_b))

        acc, n_a, n_b, _, f_a, f_b = collect_eye_features(
            model_a, model_b, path_a, path_b, cohort,
            CLASS_ORDER_EYE, list_a, list_b, args.eye_img_size,
            args.eye_is_augment, device1, device2,
        )
        print(f"{cohort}: eye accuracy = {100 * acc / len(list_a):.2f}%")

        n_a_dedup, f_a_dedup = dedup_by_subject_key(n_a, f_a)
        n_b_dedup, f_b_dedup = dedup_by_subject_key(n_b, f_b)
        name_a_all.extend(n_a_dedup)
        feat_a_all.extend(f_a_dedup)
        name_b_all.extend(n_b_dedup)
        feat_b_all.extend(f_b_dedup)
        label_all.extend([cohort_idx] * len(n_a_dedup))

    feat_a_all = reorder_list_by_ids(feat_a_all, name_a_all, args.subject_order)
    feat_b_all = reorder_list_by_ids(feat_b_all, name_a_all, args.subject_order)

    feat_a = np.array(feat_a_all).squeeze(axis=1)
    feat_b = np.array(feat_b_all).squeeze(axis=1)
    # Permute logit columns so the class order matches the tabular models'
    # (8W, 3M, 12M, 18M). The originally trained eye heads output
    # (12M, 18M, 3M, 8W), so we remap [0, 1, 2, 3] -> [3, 2, 0, 1].
    feat_a[:, [0, 1, 2, 3]] = feat_a[:, [3, 2, 0, 1]]
    feat_b[:, [0, 1, 2, 3]] = feat_b[:, [3, 2, 0, 1]]

    # --- Behaviour modality -----------------------------------------------
    behavior_df = pd.read_csv(args.behavior_test_csv)
    behavior_df = reorder_df_by_ids(
        behavior_df, behavior_df["id"].tolist(), args.subject_order
    )
    behavior_df = behavior_df.drop(columns=["id"])
    y_true_behavior = behavior_df["label"].values
    behavior_features = behavior_df.drop(columns=["label"])
    beh_cols = [f"feature{i + 1}" for i in range(args.behavior_n_features)]
    _, feat_behavior = predict_tabular(
        behavior_features, beh_cols,
        os.path.join(args.behavior_model_dir,
                     f"model-{args.behavior_classifier}.pkl"),
        os.path.join(args.behavior_model_dir,
                     f"imputer-{args.behavior_classifier}.pkl"),
        os.path.join(args.behavior_model_dir,
                     f"scaler-{args.behavior_classifier}.pkl"),
    )

    # --- Gait modality ----------------------------------------------------
    gait_df = pd.read_csv(args.gait_test_csv)
    gait_ids = gait_df["id"].values
    gait_df = gait_df.drop(columns=["id"])
    gait_features = gait_df.drop(columns=["label"])
    _, gait_probs = predict_tabular(
        gait_features, args.gait_feature_cols,
        os.path.join(args.gait_model_dir,
                     f"model-{args.gait_classifier}.pkl"),
        os.path.join(args.gait_model_dir,
                     f"imputer-{args.gait_classifier}.pkl"),
        os.path.join(args.gait_model_dir,
                     f"scaler-{args.gait_classifier}.pkl"),
    )
    _, feat_gait, y_true_gait = aggregate_by_subject(
        gait_ids, gait_probs, gait_df["label"].values
    )

    # Sanity check: all three modalities should agree on the subject labels
    y_true = y_true_gait  # subject-aligned ground truth
    assert np.array_equal(y_true, y_true_behavior), (
        "Subject-level labels disagree across modalities."
    )

    # --- Fusion variants --------------------------------------------------
    w = args.use_fixed_weights
    fusions = {
        "single_feature_predict_eye_A":
            fuse_single(feat_a),
        "single_feature_predict_eye_B":
            fuse_single(feat_b),
        "cross_feature_predict_eye":
            fuse_two_eye(feat_a, feat_b, w),
        "cross_feature_predict_eye_gait":
            fuse_eye_gait(feat_a, feat_b, feat_gait, w),
        "cross_feature_predict_eye_behavior":
            fuse_eye_behavior(feat_a, feat_b, feat_behavior, w),
        "cross_feature_predict_gait_behavior":
            fuse_gait_behavior(feat_gait, feat_behavior, w),
        "cross_feature_predict_all":
            fuse_all(feat_a, feat_b, feat_gait, feat_behavior, w),
    }

    all_results: List[Dict[str, float]] = []
    for key, fused in fusions.items():
        metrics = evaluate_fusion(
            key, np.asarray(fused), y_true,
            args.output_feature_dir, args.eye_arch,
            args.eye_img_size, args.eye_epochs,
        )
        all_results.append(metrics)
        print(f"Evaluated: {METHOD_LABELS[key]}  acc={metrics['Accuracy']:.3f}")

    save_report(all_results, args.output_report_xlsx)
    print(f"\nReport written to {args.output_report_xlsx}")


if __name__ == "__main__":
    main()
