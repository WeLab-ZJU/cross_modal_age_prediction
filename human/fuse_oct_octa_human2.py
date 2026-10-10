#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fuse_oct_octa_human.py

Human OCT / OCTA dual-modality fusion inference
with classification metrics + three MAE metrics.

============================================================
Data layout
============================================================

data/
├── OCT/
│   └── test/
│       ├── 1-30/
│       ├── 30-60/
│       ├── 60-80/
│       └── 80+/
└── OCTA/
    └── test/
        ├── 1-30/
        ├── 30-60/
        ├── 60-80/
        └── 80+/

For example:

--oct_test_root ./data/OCT/test
--octa_test_root ./data/OCTA/test


============================================================
Subject aggregation
============================================================

The following views of a subject:

    OCT_L
    OCT_R
    OCT_L_elastic
    OCT_R_vflip
    OCT_L_hflip
    ...

are all treated as belonging to that same subject.

Run inference for each view, then:

    subject_logits = mean(view_logits)


============================================================
OCT + OCTA entropy fusion
============================================================

For OCT/OCTA:

1. logits -> row_normalize_l2
2. Compute entropy
3. Obtain two modality weights using softmax(-entropy)
4. Compute weighted fusion
5. Take argmax for the final class


============================================================
Representative ages for each class
============================================================

By default, derive values from class_order:

    1-30   -> 15.5
    30-60  -> 45
    60-80  -> 70
    80+    -> 80

In particular:

    The 80+ class uses a fixed representative age of 80 years.


============================================================
MAE-1
============================================================

Calculate per subject:

    MAE-1 =
        mean(
            |mid[pred_class] - mid[true_class]|
        )

For example:

    true = 1-30
    pred = 30-60

    error =
        |15.5 - 45|
        = 29.5


============================================================
MAE-2
============================================================

Calculate per image.

Extract the true age from the filename:

    93136128_29y_FP_L.jpg
             ^^
             real_age = 29

Then:

    MAE-2 =
        mean(
            |mid[pred_class] - real_age|
        )

Note:

Model predictions remain subject-level predictions.
All images of one subject share the same subject-level prediction,
but the true age is read separately from each image filename.


============================================================
MAE-3
============================================================

First estimate continuous age using the full class-probability distribution.

For example:

    class_age =
        [15.5, 45, 70, 80]

    probability =
        [0.20, 0.60, 0.15, 0.05]

Predicted age:

    pred_age =
        15.5 * 0.20
        + 45 * 0.60
        + 70 * 0.15
        + 80 * 0.05

        = 44.6

Then compare with the true age of each image:

    MAE-3 =
        mean(
            |pred_age - real_age|
        )

This method is mathematically equivalent to:

    base_age + probability weighted correction



============================================================
Outputs
============================================================

output_dir/
├── predictions.csv
├── metrics.csv
├── mae_metrics.csv
├── mae_subject_details.csv
├── mae_image_details.csv
├── skipped_samples.csv
└── fusion_report.xlsx


metrics.csv:
    Classification metrics and three MAE metrics

mae_metrics.csv:
    Separate summary of the three MAE metrics

mae_subject_details.csv:
    Subject-level calculation details for MAE-1

mae_image_details.csv:
    Per-image calculation details for MAE-2 / MAE-3


============================================================
Dependencies
============================================================

pip install torch torchvision timm pandas openpyxl pillow scikit-learn numpy
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import models_vit


# ============================================================
# Basic configuration
# ============================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

DEFAULT_CLASS_ORDER = (
    "1-30",
    "30-60",
    "60-80",
    "80+",
)

DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Human OCT/OCTA entropy fusion "
            "with classification metrics and MAE."
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--oct_arch",
        default="vit_base_patch16",
        choices=[
            "vit_base_patch16",
            "vit_large_patch16",
            "vit_huge_patch14",
        ],
    )

    parser.add_argument(
        "--octa_arch",
        default="vit_base_patch16",
        choices=[
            "vit_base_patch16",
            "vit_large_patch16",
            "vit_huge_patch14",
        ],
    )

    parser.add_argument(
        "--oct_ckpt",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--octa_ckpt",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--checkpoint_key",
        default="model",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--oct_test_root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--octa_test_root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--class_order",
        nargs="+",
        default=list(DEFAULT_CLASS_ORDER),
    )

    parser.add_argument(
        "--class_age_values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Representative age of each class. "
            "Example: "
            "--class_age_values 15.5 45 70 80"
        ),
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    parser.add_argument(
        "--oct_img_size",
        type=int,
        default=196,
    )

    parser.add_argument(
        "--octa_img_size",
        type=int,
        default=196,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    parser.add_argument(
        "--oct_device",
        default="cuda:0",
    )

    parser.add_argument(
        "--octa_device",
        default="cuda:1",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/human_oct_octa_fusion"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Basic utilities
# ============================================================

def normalize_extensions(
    exts: Iterable[str],
) -> Tuple[str, ...]:

    return tuple(
        e.lower()
        if e.startswith(".")
        else f".{e.lower()}"
        for e in exts
    )


def get_device(
    name: str,
) -> torch.device:

    if (
        name.startswith("cuda")
        and not torch.cuda.is_available()
    ):

        print(
            f"[WARN] CUDA unavailable. "
            f"Fallback {name} -> cpu"
        )

        return torch.device("cpu")

    return torch.device(name)


# ============================================================
# checkpoint
# ============================================================

def strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    if not any(
        k.startswith("module.")
        for k in state_dict
    ):
        return state_dict

    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


def extract_state_dict(
    checkpoint: object,
    checkpoint_key: str,
) -> Dict[str, torch.Tensor]:

    if isinstance(checkpoint, dict):

        if (
            checkpoint_key in checkpoint
            and isinstance(
                checkpoint[checkpoint_key],
                dict,
            )
        ):
            return checkpoint[checkpoint_key]

        tensor_values = [
            v
            for v in checkpoint.values()
            if torch.is_tensor(v)
        ]

        if tensor_values:
            return checkpoint

    raise KeyError(
        "Cannot find model state_dict. "
        f"checkpoint_key='{checkpoint_key}'"
    )


def build_model(
    checkpoint_path: Path,
    arch: str,
    img_size: int,
    num_classes: int,
    checkpoint_key: str,
    device: torch.device,
) -> torch.nn.Module:

    if arch not in models_vit.__dict__:

        raise KeyError(
            f"Unknown model architecture: {arch}"
        )

    model = models_vit.__dict__[arch](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=0.0,
        global_pool=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = strip_module_prefix(
        extract_state_dict(
            checkpoint,
            checkpoint_key,
        )
    )

    # Check the classification head
    for key in (
        "head.weight",
        "head.bias",
    ):

        if (
            key in state_dict
            and key in model.state_dict()
        ):

            if (
                tuple(state_dict[key].shape)
                !=
                tuple(
                    model.state_dict()[key].shape
                )
            ):

                raise RuntimeError(
                    f"{checkpoint_path}: "
                    f"{key} shape mismatch: "
                    f"{tuple(state_dict[key].shape)} "
                    f"vs "
                    f"{tuple(model.state_dict()[key].shape)}"
                )

    msg = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"[INFO] Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"[INFO] Missing keys: "
        f"{len(msg.missing_keys)}, "
        f"unexpected keys: "
        f"{len(msg.unexpected_keys)}"
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# Images
# ============================================================

def load_and_normalize(
    path: Path,
    img_size: int,
) -> np.ndarray:

    img = Image.open(path)

    img = img.convert("RGB")

    img = img.resize(
        (img_size, img_size),
        Image.Resampling.BICUBIC,
    )

    arr = (
        np.asarray(img)
        .astype(np.float32)
        / 255.0
    )

    arr = (
        arr - IMAGENET_MEAN
    ) / IMAGENET_STD

    return arr


def image_to_tensor(
    images: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:

    arr = np.stack(
        images,
        axis=0,
    )

    x = (
        torch.from_numpy(arr)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    return (
        x.to(
            device,
            non_blocking=True,
        )
        .float()
    )


# ============================================================
# Numerical utilities
# ============================================================

def softmax_np(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = (
        x
        - np.max(
            x,
            axis=axis,
            keepdims=True,
        )
    )

    exp_x = np.exp(x)

    return (
        exp_x
        /
        np.sum(
            exp_x,
            axis=axis,
            keepdims=True,
        )
    )


def row_normalize_l2(
    mat: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:

    mat = np.asarray(
        mat,
        dtype=np.float64,
    )

    if mat.ndim != 2:

        raise ValueError(
            "Input must be a 2-D matrix."
        )

    row_sums = mat.sum(
        axis=1,
        keepdims=True,
    )

    needs_softmax = ~np.isclose(
        row_sums,
        1.0,
        atol=1e-5,
    )

    processed = mat.copy()

    for i in range(
        mat.shape[0]
    ):

        if needs_softmax[i, 0]:

            processed[i] = softmax_np(
                mat[i][None, :],
                axis=1,
            )[0]

    norms = np.linalg.norm(
        processed,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    return (
        processed
        /
        (
            norms
            + eps
        )
    )


def calculate_entropy(
    feature: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:

    return -np.sum(
        feature
        *
        np.log(
            feature + eps
        ),
        axis=1,
    )


def normalize_score_distribution(
    score: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert nonnegative scores to probabilities whose rows sum to 1.
    """

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    if score.ndim != 2:

        raise ValueError(
            "score must be 2-D."
        )

    # Fusion scores should already be nonnegative;
    # also handle floating-point errors here.
    score = np.maximum(
        score,
        0.0,
    )

    row_sum = score.sum(
        axis=1,
        keepdims=True,
    )

    zero_rows = (
        row_sum[:, 0]
        <= eps
    )

    if np.any(zero_rows):

        score = score.copy()

        score[
            zero_rows
        ] = (
            1.0
            /
            score.shape[1]
        )

        row_sum = score.sum(
            axis=1,
            keepdims=True,
        )

    return (
        score
        /
        row_sum
    )


# ============================================================
# Representative class ages
# ============================================================

_CLASS_RANGE_RE = re.compile(
    r"^\s*"
    r"(?P<low>\d+(?:\.\d+)?)"
    r"\s*-\s*"
    r"(?P<high>\d+(?:\.\d+)?)"
    r"\s*$"
)

_CLASS_PLUS_RE = re.compile(
    r"^\s*"
    r"(?P<low>\d+(?:\.\d+)?)"
    r"\s*\+\s*$"
)


def infer_class_age_values(
    class_order: Sequence[str],
) -> np.ndarray:

    result = []

    for class_name in class_order:

        class_name = str(
            class_name
        )

        m = _CLASS_RANGE_RE.match(
            class_name
        )

        if m:

            low = float(
                m.group("low")
            )

            high = float(
                m.group("high")
            )

            result.append(
                (
                    low + high
                )
                / 2.0
            )

            continue

        m = _CLASS_PLUS_RE.match(
            class_name
        )

        if m:

            # 80+ -> 80
            result.append(
                float(
                    m.group("low")
                )
            )

            continue

        raise ValueError(
            "Cannot infer representative age "
            f"from class '{class_name}'. "
            "Please use --class_age_values."
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


# ============================================================
# True age from filenames
# ============================================================

# Supported patterns:
#
# 93136128_29y_FP_L
# 93136128_29Y_FP_R
# elastic-93136128_29y_OCT_L
# 10536546_74.5y_OCTA_R
#
# An age token may be preceded by:
#   Start of string / "_" / "-"
#
# An age token may be followed by:
#   End of string / "_" / "-"
#
_AGE_TOKEN_RE = re.compile(
    r"(?:^|[_-])"
    r"(?P<age>\d{1,3}(?:\.\d+)?)"
    r"[yY]"
    r"(?=[_-]|$)"
)


def extract_real_age_from_path(
    path: Path,
) -> float:

    m = _AGE_TOKEN_RE.search(
        path.stem
    )

    if m is None:
        return float("nan")

    return float(
        m.group("age")
    )


def extract_real_age_from_files(
    files: Sequence[Path],
) -> float:
    """
    Provide a subject's true age for display in predictions.csv.

    MAE-2/3 do not depend on this function;
    they extract the true age again for each image.
    """

    ages = []

    for path in files:

        age = extract_real_age_from_path(
            path
        )

        if np.isfinite(age):
            ages.append(age)

    if not ages:
        return float("nan")

    unique_ages = np.unique(
        np.asarray(
            ages,
            dtype=np.float64,
        )
    )

    if len(unique_ages) > 1:

        print(
            "[WARN] Different ages detected "
            "inside one subject:"
        )

        print(
            [p.name for p in files]
        )

        print(
            "ages =",
            unique_ages.tolist(),
        )

        return float("nan")

    return float(
        unique_ages[0]
    )


# ============================================================
# MAE-3 probability-based age
# ============================================================

def estimate_age_from_probability(
    score: np.ndarray,
    class_age_values: np.ndarray,
) -> np.ndarray:
    """
    Calculate continuous predicted age from the full probability distribution.

    Given:

        class_age_values =
            [15.5, 45, 70, 80]

        prob =
            [p0, p1, p2, p3]

    Obtain:

        predicted_age =
            p0 * 15.5
            + p1 * 45
            + p2 * 70
            + p3 * 80

    This is mathematically equivalent to:

        base_age
        + probability weighted correction

    """

    prob = normalize_score_distribution(
        score
    )

    class_age_values = np.asarray(
        class_age_values,
        dtype=np.float64,
    )

    if (
        prob.shape[1]
        !=
        len(class_age_values)
    ):

        raise ValueError(
            "Number of score classes "
            f"{prob.shape[1]} "
            "does not match age values "
            f"{len(class_age_values)}."
        )

    return (
        prob
        @ class_age_values
    )


# ============================================================
# Model predictions
# ============================================================

def predict_file_group(
    model: torch.nn.Module,
    files: Sequence[Path],
    img_size: int,
    device: torch.device,
) -> np.ndarray:

    images = [
        load_and_normalize(
            path,
            img_size,
        )
        for path in files
    ]

    x = image_to_tensor(
        images,
        device,
    )

    with torch.no_grad():

        logits = (
            model(x)
            .detach()
            .cpu()
            .numpy()
        )

    # Average logits over all views of the same subject
    return logits.mean(
        axis=0
    )


# ============================================================
# Subject and L/R parsing from filenames
# ============================================================

_SIDE_SUFFIX_RE = re.compile(
    r"^(?P<base>.+)_"
    r"(?P<side>[LR])$",
    flags=re.IGNORECASE,
)

_AUG_PREFIXES = (
    "elastic",
    "flip1",
    "flip2",
    "origin",
    "zoom",
)

_AUG_SUFFIXES = (
    "_elastic",
    "_vflip",
    "_hflip",
    "_zoom",
    "_flip1",
    "_flip2",
    "_origin",
)

_MOD_TOKEN_RE = re.compile(
    r"_(OCT|OCTA)(?=_|$)",
    flags=re.IGNORECASE,
)


def split_subject_side(
    path: Path,
) -> Tuple[str, Optional[str]]:

    stem = path.stem

    # --------------------------------------------------------
    # Remove augmentation prefixes
    # --------------------------------------------------------

    if "-" in stem:

        prefix, remaining = stem.split(
            "-",
            1,
        )

        if (
            prefix.lower()
            in _AUG_PREFIXES
        ):
            stem = remaining

    # --------------------------------------------------------
    # Remove augmentation suffixes
    # --------------------------------------------------------

    for suffix in _AUG_SUFFIXES:

        if stem.lower().endswith(
            suffix.lower()
        ):

            stem = stem[
                : -len(suffix)
            ]

            break

    # --------------------------------------------------------
    # Remove OCT / OCTA tokens
    # --------------------------------------------------------

    stem = _MOD_TOKEN_RE.sub(
        "",
        stem,
    )

    # --------------------------------------------------------
    # Remove L / R markers
    # --------------------------------------------------------

    m = _SIDE_SUFFIX_RE.match(
        stem
    )

    if m:

        return (
            m.group("base"),
            m.group("side").upper(),
        )

    return stem, None


# ============================================================
# Data files
# ============================================================

def list_images(
    class_dir: Path,
    extensions: Tuple[str, ...],
    recursive: bool,
) -> List[Path]:

    if not class_dir.exists():

        raise FileNotFoundError(
            f"Class folder does not exist: "
            f"{class_dir}"
        )

    iterator = (
        class_dir.rglob("*")
        if recursive
        else class_dir.glob("*")
    )

    files = [
        path
        for path in iterator
        if (
            path.is_file()
            and
            path.suffix.lower()
            in extensions
        )
    ]

    return sorted(
        files,
        key=lambda p: p.name,
    )


def group_lr_pairs(
    files: Sequence[Path],
) -> List[Dict[str, object]]:

    buckets: Dict[
        str,
        Dict[str, List[Path]]
    ] = defaultdict(
        lambda: {
            "L": [],
            "R": [],
            "X": [],
        }
    )

    for path in files:

        subject_id, side = (
            split_subject_side(
                path
            )
        )

        if side in (
            "L",
            "R",
        ):

            buckets[
                subject_id
            ][side].append(
                path
            )

        else:

            buckets[
                subject_id
            ]["X"].append(
                path
            )

    groups = []

    for subject_id in sorted(
        buckets.keys()
    ):

        b = buckets[
            subject_id
        ]

        files_all = (
            b["L"]
            + b["R"]
            + b["X"]
        )

        if not files_all:
            continue

        has_l = bool(b["L"])
        has_r = bool(b["R"])

        if has_l and has_r:

            paired = True
            side_info = "L+R"

        elif has_l:

            paired = False
            side_info = "L_only"

        elif has_r:

            paired = False
            side_info = "R_only"

        else:

            paired = False
            side_info = "single"

        groups.append(
            {
                "subject_id": subject_id,
                "files": files_all,
                "paired": paired,
                "side_info": side_info,
            }
        )

    return groups


# ============================================================
# Single-modality inference
# ============================================================

def run_modality_inference(
    modality_name: str,
    model: torch.nn.Module,
    test_root: Path,
    class_order: Sequence[str],
    img_size: int,
    device: torch.device,
    extensions: Tuple[str, ...],
    recursive: bool,
) -> Dict[str, Dict[str, object]]:

    records = {}

    for label_idx, label_name in enumerate(
        class_order
    ):

        class_dir = (
            test_root
            / label_name
        )

        files = list_images(
            class_dir,
            extensions,
            recursive,
        )

        groups = group_lr_pairs(
            files
        )

        paired_count = sum(
            bool(g["paired"])
            for g in groups
        )

        print(
            f"[{modality_name}] "
            f"{label_name}: "
            f"{len(files)} files -> "
            f"{len(groups)} subjects, "
            f"{paired_count} paired L/R"
        )

        for group in groups:

            sid = str(
                group["subject_id"]
            )

            group_files = list(
                group["files"]
            )

            if sid in records:

                print(
                    "[WARN] subject duplicated "
                    "across classes:"
                )

                print(
                    f"  modality={modality_name}"
                )

                print(
                    f"  subject={sid}"
                )

                print(
                    "  old="
                    f"{records[sid]['label_name']}"
                )

                print(
                    f"  new={label_name}"
                )

                continue

            logits = predict_file_group(
                model=model,
                files=group_files,
                img_size=img_size,
                device=device,
            )

            real_age = (
                extract_real_age_from_files(
                    group_files
                )
            )

            records[
                sid
            ] = {
                "label_idx": label_idx,
                "label_name": label_name,
                "real_age": real_age,
                "score": logits,
                "files": ";".join(
                    str(path)
                    for path in group_files
                ),
                "paired": bool(
                    group["paired"]
                ),
                "side_info": str(
                    group["side_info"]
                ),
            }

    return records


# ============================================================
# Entropy fusion
# ============================================================

def fuse_scores_entropy(
    oct_logits: np.ndarray,
    octa_logits: np.ndarray,
) -> np.ndarray:

    f_oct = row_normalize_l2(
        oct_logits
    )

    f_octa = row_normalize_l2(
        octa_logits
    )

    e_oct = calculate_entropy(
        f_oct
    )

    e_octa = calculate_entropy(
        f_octa
    )

    # Lower entropy -> larger -entropy -> greater weight
    entropy_score = np.stack(
        [
            -e_oct,
            -e_octa,
        ],
        axis=1,
    )

    weights = softmax_np(
        entropy_score,
        axis=1,
    )

    fused = (
        f_oct
        * weights[:, 0:1]
        +
        f_octa
        * weights[:, 1:2]
    )

    return fused


# ============================================================
# Classification metrics
# ============================================================

def safe_auc(
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:

    try:

        return float(
            roc_auc_score(
                y_true,
                score,
                multi_class="ovr",
                average="macro",
            )
        )

    except Exception as exc:

        print(
            "[WARN] AUC failed:",
            exc,
        )

        return float("nan")


def compute_classification_metrics(
    name: str,
    y_true: np.ndarray,
    score: np.ndarray,
) -> Dict[str, object]:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    score = normalize_score_distribution(
        score
    )

    y_pred = score.argmax(
        axis=1
    )

    return {
        "method": name,

        "n": int(
            len(y_true)
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_auc_ovr": safe_auc(
            y_true,
            score,
        ),
    }


# ============================================================
# MAE utilities
# ============================================================

def split_file_string(
    files_str: object,
) -> List[Path]:

    if not isinstance(
        files_str,
        str,
    ):
        return []

    if not files_str.strip():
        return []

    return [
        Path(item)
        for item in files_str.split(";")
        if item.strip()
    ]


def calculate_mae_results(
    pred_df: pd.DataFrame,
    class_age_values: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate three MAE metrics.

    ----------------------------------------------------------
    MAE-1:
        Subject level

    MAE-2:
        Image level

    MAE-3:
        Image level

    Returns:
        mae_metrics_df
        mae_subject_details_df
        mae_image_details_df
    """

    class_age_values = np.asarray(
        class_age_values,
        dtype=np.float64,
    )

    summary_rows = []

    subject_detail_rows = []

    image_detail_rows = []

    # --------------------------------------------------------
    # Method configuration
    # --------------------------------------------------------

    method_configs = [
        {
            "method": "OCT",
            "pred_idx_col": "oct_pred_idx",
            "pred_name_col": "oct_pred_name",
            "pred_age_mid_col": "oct_pred_age_mid",
            "pred_age_prob_col": "oct_pred_age_prob",
            "file_columns": [
                ("OCT", "oct_files"),
            ],
            "fusion_only": False,
        },

        {
            "method": "OCTA",
            "pred_idx_col": "octa_pred_idx",
            "pred_name_col": "octa_pred_name",
            "pred_age_mid_col": "octa_pred_age_mid",
            "pred_age_prob_col": "octa_pred_age_prob",
            "file_columns": [
                ("OCTA", "octa_files"),
            ],
            "fusion_only": False,
        },

        {
            "method": "OCT+OCTA(entropy)",
            "pred_idx_col": "fusion_pred_idx",
            "pred_name_col": "fusion_pred_name",
            "pred_age_mid_col": "fusion_pred_age_mid",
            "pred_age_prob_col": "fusion_pred_age_prob",

            # For subjects genuinely matched across both modalities,
            # expand the images from both OCT and OCTA individually.
            "file_columns": [
                ("OCT", "oct_files"),
                ("OCTA", "octa_files"),
            ],

            # Include only genuine entropy fusion;
            # exclude OCT_only/OCTA_only fallback predictions from dual-modality MAE.
            "fusion_only": True,
        },
    ]

    for config in method_configs:

        method = config["method"]

        pred_idx_col = config[
            "pred_idx_col"
        ]

        pred_name_col = config[
            "pred_name_col"
        ]

        pred_age_mid_col = config[
            "pred_age_mid_col"
        ]

        pred_age_prob_col = config[
            "pred_age_prob_col"
        ]

        # ----------------------------------------------------
        # Select valid subjects for this method
        # ----------------------------------------------------

        current = pred_df.copy()

        current = current[
            current[pred_idx_col] >= 0
        ]

        if config["fusion_only"]:

            current = current[
                current["fusion_type"]
                == "entropy"
            ]

        # ====================================================
        # MAE-1: SUBJECT level
        # ====================================================

        mae1_errors = []

        for _, row in current.iterrows():

            true_idx = int(
                row["true_label_idx"]
            )

            pred_idx = int(
                row[pred_idx_col]
            )

            true_age_mid = float(
                class_age_values[
                    true_idx
                ]
            )

            pred_age_mid = float(
                class_age_values[
                    pred_idx
                ]
            )

            error = abs(
                pred_age_mid
                - true_age_mid
            )

            mae1_errors.append(
                error
            )

            subject_detail_rows.append(
                {
                    "method": method,

                    "subject_id":
                        row["subject_id"],

                    "true_label_idx":
                        true_idx,

                    "true_label_name":
                        row["true_label_name"],

                    "pred_label_idx":
                        pred_idx,

                    "pred_label_name":
                        row[pred_name_col],

                    "true_age_mid":
                        true_age_mid,

                    "pred_age_mid":
                        pred_age_mid,

                    "mae_1_abs_error":
                        error,
                }
            )

        mae1 = (
            float(
                np.mean(
                    mae1_errors
                )
            )
            if mae1_errors
            else float("nan")
        )

        # ====================================================
        # MAE-2 / MAE-3: IMAGE level
        # ====================================================

        mae2_errors = []
        mae3_errors = []

        total_image_n = 0
        valid_age_image_n = 0
        invalid_age_image_n = 0

        for _, row in current.iterrows():

            true_idx = int(
                row["true_label_idx"]
            )

            pred_idx = int(
                row[pred_idx_col]
            )

            true_age_mid = float(
                class_age_values[
                    true_idx
                ]
            )

            pred_age_mid = float(
                row[
                    pred_age_mid_col
                ]
            )

            pred_age_prob = float(
                row[
                    pred_age_prob_col
                ]
            )

            for (
                source_modality,
                file_column,
            ) in config["file_columns"]:

                files = split_file_string(
                    row.get(
                        file_column,
                        "",
                    )
                )

                for path in files:

                    total_image_n += 1

                    real_age = (
                        extract_real_age_from_path(
                            path
                        )
                    )

                    if np.isfinite(
                        real_age
                    ):

                        valid_age_image_n += 1

                        mae2_error = abs(
                            pred_age_mid
                            - real_age
                        )

                        mae3_error = abs(
                            pred_age_prob
                            - real_age
                        )

                        mae2_errors.append(
                            mae2_error
                        )

                        mae3_errors.append(
                            mae3_error
                        )

                    else:

                        invalid_age_image_n += 1

                        mae2_error = np.nan
                        mae3_error = np.nan

                    image_detail_rows.append(
                        {
                            "method": method,

                            "subject_id":
                                row["subject_id"],

                            "source_modality":
                                source_modality,

                            "image_path":
                                str(path),

                            "image_name":
                                path.name,

                            "true_label_idx":
                                true_idx,

                            "true_label_name":
                                row[
                                    "true_label_name"
                                ],

                            "pred_label_idx":
                                pred_idx,

                            "pred_label_name":
                                row[
                                    pred_name_col
                                ],

                            "true_age_mid":
                                true_age_mid,

                            "real_age":
                                real_age,

                            "pred_age_mid":
                                pred_age_mid,

                            "pred_age_prob":
                                pred_age_prob,

                            "mae_2_abs_error":
                                mae2_error,

                            "mae_3_abs_error":
                                mae3_error,
                        }
                    )

        mae2 = (
            float(
                np.mean(
                    mae2_errors
                )
            )
            if mae2_errors
            else float("nan")
        )

        mae3 = (
            float(
                np.mean(
                    mae3_errors
                )
            )
            if mae3_errors
            else float("nan")
        )

        summary_rows.append(
            {
                "method": method,

                # --------------------------------------------
                # MAE 1
                # --------------------------------------------

                "mae_1_class_mid":
                    mae1,

                "mae_1_subject_n":
                    int(
                        len(
                            mae1_errors
                        )
                    ),

                # --------------------------------------------
                # MAE 2
                # --------------------------------------------

                "mae_2_real_age_midpred":
                    mae2,

                # --------------------------------------------
                # MAE 3
                # --------------------------------------------

                "mae_3_real_age_probpred":
                    mae3,

                # --------------------------------------------
                # Image count for MAE2/3
                # --------------------------------------------

                "mae_total_image_n":
                    int(
                        total_image_n
                    ),

                "mae_real_age_image_n":
                    int(
                        valid_age_image_n
                    ),

                "mae_age_parse_failed_image_n":
                    int(
                        invalid_age_image_n
                    ),
            }
        )

    mae_metrics_df = pd.DataFrame(
        summary_rows
    )

    mae_subject_details_df = (
        pd.DataFrame(
            subject_detail_rows
        )
    )

    mae_image_details_df = (
        pd.DataFrame(
            image_detail_rows
        )
    )

    return (
        mae_metrics_df,
        mae_subject_details_df,
        mae_image_details_df,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # Argument validation
    # ========================================================

    if (
        len(args.class_order)
        !=
        args.num_classes
    ):

        raise ValueError(
            "--class_order length "
            f"{len(args.class_order)} "
            "must equal "
            f"--num_classes {args.num_classes}"
        )

    if (
        args.output_dir.exists()
        and not args.overwrite
    ):

        existing_files = [
            args.output_dir
            / "predictions.csv",

            args.output_dir
            / "metrics.csv",

            args.output_dir
            / "mae_metrics.csv",
        ]

        if any(
            path.exists()
            for path in existing_files
        ):

            raise FileExistsError(
                f"{args.output_dir} "
                "already contains results. "
                "Use --overwrite."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extensions = normalize_extensions(
        args.extensions
    )

    # ========================================================
    # Representative class ages
    # ========================================================

    if args.class_age_values is None:

        class_age_values = (
            infer_class_age_values(
                args.class_order
            )
        )

    else:

        if (
            len(args.class_age_values)
            !=
            args.num_classes
        ):

            raise ValueError(
                "--class_age_values length "
                "must equal --num_classes."
            )

        class_age_values = np.asarray(
            args.class_age_values,
            dtype=np.float64,
        )

    print()
    print("=" * 70)

    print(
        "[INFO] Class order:"
    )

    for idx, (
        class_name,
        age,
    ) in enumerate(
        zip(
            args.class_order,
            class_age_values,
        )
    ):

        print(
            f"  [{idx}] "
            f"{class_name} "
            f"-> {age:.2f} years"
        )

    print("=" * 70)
    print()

    # ========================================================
    # Device
    # ========================================================

    oct_device = get_device(
        args.oct_device
    )

    octa_device = get_device(
        args.octa_device
    )

    # ========================================================
    # Models
    # ========================================================

    oct_model = build_model(
        checkpoint_path=args.oct_ckpt,
        arch=args.oct_arch,
        img_size=args.oct_img_size,
        num_classes=args.num_classes,
        checkpoint_key=args.checkpoint_key,
        device=oct_device,
    )

    octa_model = build_model(
        checkpoint_path=args.octa_ckpt,
        arch=args.octa_arch,
        img_size=args.octa_img_size,
        num_classes=args.num_classes,
        checkpoint_key=args.checkpoint_key,
        device=octa_device,
    )

    # ========================================================
    # Inference
    # ========================================================

    oct_records = run_modality_inference(
        modality_name="OCT",
        model=oct_model,
        test_root=args.oct_test_root,
        class_order=args.class_order,
        img_size=args.oct_img_size,
        device=oct_device,
        extensions=extensions,
        recursive=args.recursive,
    )

    octa_records = run_modality_inference(
        modality_name="OCTA",
        model=octa_model,
        test_root=args.octa_test_root,
        class_order=args.class_order,
        img_size=args.octa_img_size,
        device=octa_device,
        extensions=extensions,
        recursive=args.recursive,
    )

    # ========================================================
    # Subject alignment
    # ========================================================

    oct_ids = set(
        oct_records.keys()
    )

    octa_ids = set(
        octa_records.keys()
    )

    common_ids = sorted(
        oct_ids
        &
        octa_ids
    )

    oct_only_ids = sorted(
        oct_ids
        -
        octa_ids
    )

    octa_only_ids = sorted(
        octa_ids
        -
        oct_ids
    )

    rows = []
    skipped_rows = []

    matched_oct_logits = []
    matched_octa_logits = []
    matched_y = []

    oct_only_logits = []
    oct_only_y = []

    octa_only_logits = []
    octa_only_y = []

    # ========================================================
    # Matched OCT + OCTA
    # ========================================================

    for sid in common_ids:

        r_oct = oct_records[
            sid
        ]

        r_octa = octa_records[
            sid
        ]

        if (
            int(
                r_oct["label_idx"]
            )
            !=
            int(
                r_octa["label_idx"]
            )
        ):

            skipped_rows.append(
                {
                    "subject_id": sid,

                    "reason": (
                        "label_mismatch: "
                        f"OCT="
                        f"{r_oct['label_name']}, "
                        f"OCTA="
                        f"{r_octa['label_name']}"
                    ),
                }
            )

            continue

        matched_oct_logits.append(
            np.asarray(
                r_oct["score"],
                dtype=np.float64,
            )
        )

        matched_octa_logits.append(
            np.asarray(
                r_octa["score"],
                dtype=np.float64,
            )
        )

        matched_y.append(
            int(
                r_oct["label_idx"]
            )
        )

        oct_age = float(
            r_oct.get(
                "real_age",
                np.nan,
            )
        )

        octa_age = float(
            r_octa.get(
                "real_age",
                np.nan,
            )
        )

        if (
            np.isfinite(oct_age)
            and
            np.isfinite(octa_age)
        ):

            if np.isclose(
                oct_age,
                octa_age,
            ):
                subject_real_age = oct_age

            else:

                print(
                    "[WARN] OCT/OCTA age "
                    f"mismatch: {sid}, "
                    f"OCT={oct_age}, "
                    f"OCTA={octa_age}"
                )

                subject_real_age = np.nan

        elif np.isfinite(oct_age):

            subject_real_age = oct_age

        elif np.isfinite(octa_age):

            subject_real_age = octa_age

        else:

            subject_real_age = np.nan

        rows.append(
            {
                "subject_id": sid,

                "fusion_type":
                    "entropy",

                "true_label_idx":
                    int(
                        r_oct["label_idx"]
                    ),

                "true_label_name":
                    str(
                        r_oct["label_name"]
                    ),

                "real_age":
                    subject_real_age,

                "oct_files":
                    str(
                        r_oct["files"]
                    ),

                "oct_paired":
                    bool(
                        r_oct["paired"]
                    ),

                "oct_side_info":
                    str(
                        r_oct["side_info"]
                    ),

                "octa_files":
                    str(
                        r_octa["files"]
                    ),

                "octa_paired":
                    bool(
                        r_octa["paired"]
                    ),

                "octa_side_info":
                    str(
                        r_octa["side_info"]
                    ),
            }
        )

    # ========================================================
    # OCT only
    # ========================================================

    for sid in oct_only_ids:

        r = oct_records[
            sid
        ]

        oct_only_logits.append(
            np.asarray(
                r["score"],
                dtype=np.float64,
            )
        )

        oct_only_y.append(
            int(
                r["label_idx"]
            )
        )

        rows.append(
            {
                "subject_id": sid,

                "fusion_type":
                    "OCT_only",

                "true_label_idx":
                    int(
                        r["label_idx"]
                    ),

                "true_label_name":
                    str(
                        r["label_name"]
                    ),

                "real_age":
                    float(
                        r.get(
                            "real_age",
                            np.nan,
                        )
                    ),

                "oct_files":
                    str(
                        r["files"]
                    ),

                "oct_paired":
                    bool(
                        r["paired"]
                    ),

                "oct_side_info":
                    str(
                        r["side_info"]
                    ),

                "octa_files":
                    "",

                "octa_paired":
                    False,

                "octa_side_info":
                    "missing",
            }
        )

    # ========================================================
    # OCTA only
    # ========================================================

    for sid in octa_only_ids:

        r = octa_records[
            sid
        ]

        octa_only_logits.append(
            np.asarray(
                r["score"],
                dtype=np.float64,
            )
        )

        octa_only_y.append(
            int(
                r["label_idx"]
            )
        )

        rows.append(
            {
                "subject_id": sid,

                "fusion_type":
                    "OCTA_only",

                "true_label_idx":
                    int(
                        r["label_idx"]
                    ),

                "true_label_name":
                    str(
                        r["label_name"]
                    ),

                "real_age":
                    float(
                        r.get(
                            "real_age",
                            np.nan,
                        )
                    ),

                "oct_files":
                    "",

                "oct_paired":
                    False,

                "oct_side_info":
                    "missing",

                "octa_files":
                    str(
                        r["files"]
                    ),

                "octa_paired":
                    bool(
                        r["paired"]
                    ),

                "octa_side_info":
                    str(
                        r["side_info"]
                    ),
            }
        )

    # ========================================================
    # Counts
    # ========================================================

    n_matched = len(
        matched_y
    )

    n_oct_only = len(
        oct_only_y
    )

    n_octa_only = len(
        octa_only_y
    )

    if (
        n_matched == 0
        and
        n_oct_only == 0
        and
        n_octa_only == 0
    ):

        raise RuntimeError(
            "No valid samples found."
        )

    C = args.num_classes

    # ========================================================
    # Matched-subject probabilities
    # ========================================================

    if n_matched > 0:

        m_oct_logits = np.vstack(
            matched_oct_logits
        )

        m_octa_logits = np.vstack(
            matched_octa_logits
        )

        m_oct_prob = softmax_np(
            m_oct_logits,
            axis=1,
        )

        m_octa_prob = softmax_np(
            m_octa_logits,
            axis=1,
        )

        m_fusion_raw = (
            fuse_scores_entropy(
                m_oct_logits,
                m_octa_logits,
            )
        )

        m_fusion_prob = (
            normalize_score_distribution(
                m_fusion_raw
            )
        )

        m_y = np.asarray(
            matched_y,
            dtype=int,
        )

    else:

        m_oct_prob = np.zeros(
            (0, C)
        )

        m_octa_prob = np.zeros(
            (0, C)
        )

        m_fusion_prob = np.zeros(
            (0, C)
        )

        m_y = np.zeros(
            (0,),
            dtype=int,
        )

    # ========================================================
    # OCT only
    # ========================================================

    if n_oct_only > 0:

        o_oct_prob = softmax_np(
            np.vstack(
                oct_only_logits
            ),
            axis=1,
        )

        o_y = np.asarray(
            oct_only_y,
            dtype=int,
        )

    else:

        o_oct_prob = np.zeros(
            (0, C)
        )

        o_y = np.zeros(
            (0,),
            dtype=int,
        )

    # ========================================================
    # OCTA only
    # ========================================================

    if n_octa_only > 0:

        a_octa_prob = softmax_np(
            np.vstack(
                octa_only_logits
            ),
            axis=1,
        )

        a_y = np.asarray(
            octa_only_y,
            dtype=int,
        )

    else:

        a_octa_prob = np.zeros(
            (0, C)
        )

        a_y = np.zeros(
            (0,),
            dtype=int,
        )

    # ========================================================
    # Full prediction arrays
    #
    # row order:
    # matched
    # OCT_only
    # OCTA_only
    # ========================================================

    N = (
        n_matched
        +
        n_oct_only
        +
        n_octa_only
    )

    oct_scores_full = np.full(
        (N, C),
        np.nan,
    )

    octa_scores_full = np.full(
        (N, C),
        np.nan,
    )

    fusion_scores_full = np.full(
        (N, C),
        np.nan,
    )

    # OCT
    oct_scores_full[
        :n_matched
    ] = m_oct_prob

    oct_scores_full[
        n_matched:
        n_matched + n_oct_only
    ] = o_oct_prob

    # OCTA
    octa_scores_full[
        :n_matched
    ] = m_octa_prob

    octa_scores_full[
        n_matched + n_oct_only:
    ] = a_octa_prob

    # Fusion
    fusion_scores_full[
        :n_matched
    ] = m_fusion_prob

    # Fall back when one modality is missing
    fusion_scores_full[
        n_matched:
        n_matched + n_oct_only
    ] = o_oct_prob

    fusion_scores_full[
        n_matched + n_oct_only:
    ] = a_octa_prob

    # ========================================================
    # Prediction
    # ========================================================

    def argmax_rows(
        arr: np.ndarray,
    ) -> np.ndarray:

        result = np.full(
            arr.shape[0],
            -1,
            dtype=int,
        )

        for i in range(
            arr.shape[0]
        ):

            if not np.all(
                np.isnan(
                    arr[i]
                )
            ):

                result[i] = int(
                    np.nanargmax(
                        arr[i]
                    )
                )

        return result

    def class_name(
        idx: int,
    ) -> str:

        if idx < 0:
            return ""

        return str(
            args.class_order[
                idx
            ]
        )

    oct_pred = argmax_rows(
        oct_scores_full
    )

    octa_pred = argmax_rows(
        octa_scores_full
    )

    fusion_pred = argmax_rows(
        fusion_scores_full
    )

    # ========================================================
    # predictions.csv
    # ========================================================

    pred_df = pd.DataFrame(
        rows
    )

    pred_df[
        "oct_pred_idx"
    ] = oct_pred

    pred_df[
        "oct_pred_name"
    ] = [
        class_name(i)
        for i in oct_pred
    ]

    pred_df[
        "octa_pred_idx"
    ] = octa_pred

    pred_df[
        "octa_pred_name"
    ] = [
        class_name(i)
        for i in octa_pred
    ]

    pred_df[
        "fusion_pred_idx"
    ] = fusion_pred

    pred_df[
        "fusion_pred_name"
    ] = [
        class_name(i)
        for i in fusion_pred
    ]

    # ========================================================
    # Per-class scores
    # ========================================================

    for class_idx, cls in enumerate(
        args.class_order
    ):

        safe_cls = (
            str(cls)
            .replace(
                "+",
                "plus",
            )
            .replace(
                "/",
                "_",
            )
        )

        pred_df[
            f"oct_score_{safe_cls}"
        ] = (
            oct_scores_full[
                :,
                class_idx
            ]
        )

        pred_df[
            f"octa_score_{safe_cls}"
        ] = (
            octa_scores_full[
                :,
                class_idx
            ]
        )

        pred_df[
            f"fusion_score_{safe_cls}"
        ] = (
            fusion_scores_full[
                :,
                class_idx
            ]
        )

    # ========================================================
    # True midpoint age
    # ========================================================

    true_idx_full = (
        pred_df[
            "true_label_idx"
        ]
        .to_numpy(
            dtype=int
        )
    )

    pred_df[
        "true_age_mid"
    ] = (
        class_age_values[
            true_idx_full
        ]
    )

    # ========================================================
    # Class-based predicted ages
    # ========================================================

    def pred_mid_age(
        pred_idx: np.ndarray,
    ) -> np.ndarray:

        result = np.full(
            len(pred_idx),
            np.nan,
        )

        valid = (
            pred_idx >= 0
        )

        result[
            valid
        ] = class_age_values[
            pred_idx[
                valid
            ]
        ]

        return result

    oct_pred_age_mid = pred_mid_age(
        oct_pred
    )

    octa_pred_age_mid = pred_mid_age(
        octa_pred
    )

    fusion_pred_age_mid = pred_mid_age(
        fusion_pred
    )

    pred_df[
        "oct_pred_age_mid"
    ] = oct_pred_age_mid

    pred_df[
        "octa_pred_age_mid"
    ] = octa_pred_age_mid

    pred_df[
        "fusion_pred_age_mid"
    ] = fusion_pred_age_mid

    # ========================================================
    # Probability-based continuous ages
    # ========================================================

    def probability_age(
        score: np.ndarray,
    ) -> np.ndarray:

        result = np.full(
            score.shape[0],
            np.nan,
        )

        valid = ~np.all(
            np.isnan(score),
            axis=1,
        )

        if np.any(valid):

            result[
                valid
            ] = (
                estimate_age_from_probability(
                    score[
                        valid
                    ],
                    class_age_values,
                )
            )

        return result

    pred_df[
        "oct_pred_age_prob"
    ] = probability_age(
        oct_scores_full
    )

    pred_df[
        "octa_pred_age_prob"
    ] = probability_age(
        octa_scores_full
    )

    pred_df[
        "fusion_pred_age_prob"
    ] = probability_age(
        fusion_scores_full
    )

    # ========================================================
    # Classification metrics
    # ========================================================

    classification_metrics = []

    # OCT:
    # matched + OCT_only
    if (
        n_matched
        +
        n_oct_only
        > 0
    ):

        classification_metrics.append(
            compute_classification_metrics(
                name="OCT",

                y_true=np.concatenate(
                    [
                        m_y,
                        o_y,
                    ]
                ),

                score=np.vstack(
                    [
                        m_oct_prob,
                        o_oct_prob,
                    ]
                ),
            )
        )

    # OCTA:
    # matched + OCTA_only
    if (
        n_matched
        +
        n_octa_only
        > 0
    ):

        classification_metrics.append(
            compute_classification_metrics(
                name="OCTA",

                y_true=np.concatenate(
                    [
                        m_y,
                        a_y,
                    ]
                ),

                score=np.vstack(
                    [
                        m_octa_prob,
                        a_octa_prob,
                    ]
                ),
            )
        )

    # Fusion:
    # Evaluate only genuinely matched subjects
    if n_matched > 0:

        classification_metrics.append(
            compute_classification_metrics(
                name=(
                    "OCT+OCTA(entropy)"
                ),

                y_true=m_y,

                score=m_fusion_prob,
            )
        )

    classification_metrics_df = (
        pd.DataFrame(
            classification_metrics
        )
    )

    # ========================================================
    # Three MAE metrics
    # ========================================================

    (
        mae_metrics_df,
        mae_subject_details_df,
        mae_image_details_df,
    ) = calculate_mae_results(
        pred_df=pred_df,
        class_age_values=class_age_values,
    )

    # ========================================================
    # Combine classification metrics and MAE
    # ========================================================

    metrics_df = (
        classification_metrics_df
        .merge(
            mae_metrics_df,
            on="method",
            how="left",
        )
    )

    # ========================================================
    # skipped
    # ========================================================

    if skipped_rows:

        skipped_df = pd.DataFrame(
            skipped_rows
        )

    else:

        skipped_df = pd.DataFrame(
            columns=[
                "subject_id",
                "reason",
            ]
        )

    # ========================================================
    # Class age mapping
    # ========================================================

    class_age_df = pd.DataFrame(
        {
            "class_idx":
                np.arange(
                    len(
                        args.class_order
                    )
                ),

            "class_name":
                list(
                    args.class_order
                ),

            "representative_age":
                class_age_values,
        }
    )

    # ========================================================
    # Outputs
    # ========================================================

    pred_path = (
        args.output_dir
        / "predictions.csv"
    )

    metrics_path = (
        args.output_dir
        / "metrics.csv"
    )

    mae_metrics_path = (
        args.output_dir
        / "mae_metrics.csv"
    )

    mae_subject_path = (
        args.output_dir
        / "mae_subject_details.csv"
    )

    mae_image_path = (
        args.output_dir
        / "mae_image_details.csv"
    )

    skipped_path = (
        args.output_dir
        / "skipped_samples.csv"
    )

    report_path = (
        args.output_dir
        / "fusion_report.xlsx"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    pred_df.to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    mae_metrics_df.to_csv(
        mae_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    mae_subject_details_df.to_csv(
        mae_subject_path,
        index=False,
        encoding="utf-8-sig",
    )

    mae_image_details_df.to_csv(
        mae_image_path,
        index=False,
        encoding="utf-8-sig",
    )

    skipped_df.to_csv(
        skipped_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    with pd.ExcelWriter(
        report_path,
        engine="openpyxl",
    ) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="metrics",
            index=False,
        )

        mae_metrics_df.to_excel(
            writer,
            sheet_name="mae_metrics",
            index=False,
        )

        pred_df.to_excel(
            writer,
            sheet_name="predictions",
            index=False,
        )

        mae_subject_details_df.to_excel(
            writer,
            sheet_name="mae_subject_details",
            index=False,
        )

        mae_image_details_df.to_excel(
            writer,
            sheet_name="mae_image_details",
            index=False,
        )

        skipped_df.to_excel(
            writer,
            sheet_name="skipped",
            index=False,
        )

        class_age_df.to_excel(
            writer,
            sheet_name="class_age_mapping",
            index=False,
        )

    # ========================================================
    # Terminal summary
    # ========================================================

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        f"Subjects in predictions: "
        f"{len(pred_df)}"
    )

    print(
        f"  entropy fusion : "
        f"{n_matched}"
    )

    print(
        f"  OCT only       : "
        f"{n_oct_only}"
    )

    print(
        f"  OCTA only      : "
        f"{n_octa_only}"
    )

    print(
        f"Skipped          : "
        f"{len(skipped_df)}"
    )

    print()

    print("=" * 90)
    print("Classification + MAE Metrics")
    print("=" * 90)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()

    print("=" * 90)
    print("MAE Metrics")
    print("=" * 90)

    print(
        mae_metrics_df.to_string(
            index=False
        )
    )

    print()

    # --------------------------------------------------------
    # Age-parsing checks
    # --------------------------------------------------------

    if len(
        mae_image_details_df
    ) > 0:

        valid_real_age = int(
            mae_image_details_df[
                "real_age"
            ]
            .notna()
            .sum()
        )

        total_images = int(
            len(
                mae_image_details_df
            )
        )

        print(
            "[INFO] MAE image records:",
            total_images,
        )

        print(
            "[INFO] Successfully parsed "
            "real age:",
            valid_real_age,
        )

        print(
            "[INFO] Failed to parse age:",
            total_images
            - valid_real_age,
        )

    print()
    print("=" * 90)

    print(
        f"Predictions       : {pred_path}"
    )

    print(
        f"Metrics           : {metrics_path}"
    )

    print(
        f"MAE metrics       : {mae_metrics_path}"
    )

    print(
        f"MAE subject detail: {mae_subject_path}"
    )

    print(
        f"MAE image detail  : {mae_image_path}"
    )

    print(
        f"Skipped           : {skipped_path}"
    )

    print(
        f"Excel report      : {report_path}"
    )

    print("=" * 90)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
