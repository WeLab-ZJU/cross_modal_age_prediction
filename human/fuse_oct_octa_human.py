#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_oct_octa_human.py

Human OCT / OCTA dual-modality fusion inference.

Intended use cases
--------
1. Human age-group classification, for example:
   1-30, 30-60, 60-80, 80+

2. The test set uses the ImageFolder layout:
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

   Alternatively, pass a root containing the class directories directly:
   --oct_test_root ./data/OCT/test
   --octa_test_root ./data/OCTA/test

3. Within-modality aggregation:
   - Check each modality for _L / _R pairs within each class directory.
   - For all views of a subject within one modality (original left/right images
     and elastic/vflip/hflip/zoom variants), average logits into one subject vector.
   - predictions.csv records oct_paired and octa_paired for inspection.

4. Cross-modality fusion (entropy-weighted features; see cross_fusion.cross_feature_predict_entropy):
   - Use only the two eye-modality features (OCT and OCTA head logits).
     Other features, such as cross_fusion's feature_bt / feature_sy, are excluded.
   - Apply row_normalize_l2 to each modality's logits (softmax then L2 if the row sum is not 1).
     Compute each modality's entropy per sample and obtain weights with softmax(-entropy).
     Lower-entropy modalities receive higher weights; sum the weighted features.
     Use argmax for the final prediction.

Outputs
----
output_dir/
├── predictions.csv
├── skipped_samples.csv
├── metrics.csv
└── fusion_report.xlsx

Dependencies
----
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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CLASS_ORDER = ("1-30", "30-60", "60-80", "80+")
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse human OCT and OCTA predictions.")

    # Model / checkpoint
    parser.add_argument("--oct_arch", default="vit_base_patch16",
                        choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"])
    parser.add_argument("--octa_arch", default="vit_base_patch16",
                        choices=["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"])
    parser.add_argument("--oct_ckpt", required=True, type=Path,
                        help="Fine-tuned OCT checkpoint path.")
    parser.add_argument("--octa_ckpt", required=True, type=Path,
                        help="Fine-tuned OCTA checkpoint path.")
    parser.add_argument("--checkpoint_key", default="model",
                        help="Checkpoint key that stores model state_dict. Default: model")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of age-stage classes. Default: 4")

    # Data
    parser.add_argument("--oct_test_root", required=True, type=Path,
                        help="OCT ImageFolder test root, containing class folders.")
    parser.add_argument("--octa_test_root", required=True, type=Path,
                        help="OCTA ImageFolder test root, containing class folders.")
    parser.add_argument("--class_order", nargs="+", default=list(DEFAULT_CLASS_ORDER),
                        help="Class folder order, must match ImageFolder/training order.")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS),
                        help="Allowed image extensions.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search class folders. Usually not needed.")

    # Image size
    parser.add_argument("--oct_img_size", type=int, default=196)
    parser.add_argument("--octa_img_size", type=int, default=196)

    # Fusion: entropy-weighted feature fusion over OCT/OCTA logits only.
    # (no manual weights / score_type — weights come from per-sample entropy)

    # Device
    parser.add_argument("--oct_device", default="cuda:0",
                        help="Device for OCT model, e.g. cuda:0 or cpu.")
    parser.add_argument("--octa_device", default="cuda:1",
                        help="Device for OCTA model, e.g. cuda:1 or cpu.")

    # Output
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/human_oct_octa_fusion"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files if present.")

    return parser.parse_args()


def normalize_extensions(exts: Iterable[str]) -> Tuple[str, ...]:
    return tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA is unavailable. Fallback from {name} to cpu.")
        return torch.device("cpu")
    return torch.device(name)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def extract_state_dict(checkpoint: object, checkpoint_key: str) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if checkpoint_key in checkpoint and isinstance(checkpoint[checkpoint_key], dict):
            return checkpoint[checkpoint_key]
        # Some checkpoints are already state dictionaries.
        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if len(tensor_values) > 0:
                return checkpoint
    raise KeyError(
        f"Cannot find model state_dict. Tried checkpoint_key='{checkpoint_key}'. "
        "Please check your checkpoint format."
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
        raise KeyError(f"Unknown ViT arch: {arch}. Available key is not found in models_vit.")

    model = models_vit.__dict__[arch](
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=0.0,
        global_pool=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_state_dict(checkpoint, checkpoint_key))

    # The classifier head must have the matching class count for meaningful inference.
    for head_key in ("head.weight", "head.bias"):
        if head_key in state_dict and head_key in model.state_dict():
            if tuple(state_dict[head_key].shape) != tuple(model.state_dict()[head_key].shape):
                raise RuntimeError(
                    f"{checkpoint_path} has incompatible {head_key} shape: "
                    f"{tuple(state_dict[head_key].shape)} vs expected "
                    f"{tuple(model.state_dict()[head_key].shape)}. "
                    f"Please set --num_classes correctly or use the matching checkpoint."
                )

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded {checkpoint_path}")
    print(f"[INFO] Missing keys: {len(msg.missing_keys)}, unexpected keys: {len(msg.unexpected_keys)}")

    model.to(device)
    model.eval()
    return model


def load_and_normalize(path: Path, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def image_to_tensor(images: List[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images, axis=0)  # NHWC
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return x.to(device, non_blocking=True).float()


def row_normalize_l2(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row normalization (following cross_fusion.row_normalize_l2):
    If a row does not sum to 1, apply softmax first, then L2 normalization.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix")

    row_sums = mat.sum(axis=1, keepdims=True)
    needs_softmax = ~np.isclose(row_sums, 1.0, atol=1e-5)
    processed = mat.copy()
    for i in range(mat.shape[0]):
        if needs_softmax[i]:
            shifted = mat[i] - np.max(mat[i])
            exp_vals = np.exp(shifted)
            processed[i] = exp_vals / (exp_vals.sum() + eps)

    norms = np.linalg.norm(processed, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return processed / (norms + eps)


def calculate_entropy(feature: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Treat each feature row as a distribution: H = -sum(p * log(p)); return [n_samples]."""
    return -np.sum(feature * np.log(feature + eps), axis=1)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def predict_file_group(
    model: torch.nn.Module,
    files: Sequence[Path],
    img_size: int,
    device: torch.device,
) -> np.ndarray:
    """Average logits over all views of one subject (original L/R images and augmentations).

    Return one subject-level logit vector [num_classes] for this modality.
    This follows cross_fusion's aggregation of L/R/augmented head outputs, using a mean before normalization.
    """
    images = [load_and_normalize(p, img_size) for p in files]
    x = image_to_tensor(images, device)

    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()  # [n_views, num_classes]

    return logits.mean(axis=0)


_SIDE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<side>[LR])$", flags=re.IGNORECASE)

# Augmentation markers: OCTA-500 uses prefixes (elastic-10186); this project uses suffixes (10186_elastic).
_AUG_PREFIXES = ("elastic", "flip1", "flip2", "origin", "zoom")
_AUG_SUFFIXES = ("_elastic", "_vflip", "_hflip", "_zoom", "_flip1", "_flip2", "_origin")
# Remove modality tokens so OCT and OCTA images of the same subject share a subject_id.
_MOD_TOKEN_RE = re.compile(r"_(OCT|OCTA)(?=_|$)", flags=re.IGNORECASE)


def split_subject_side(path: Path) -> Tuple[str, Optional[str]]:
    """
    Infer subject_id and eye side from the filename for cross-modality OCT/OCTA alignment.

    Processing order:
      1. Remove augmentation prefixes (OCTA-500): elastic-10186_L -> 10186_L
      2. Remove augmentation suffixes: 10536546_74y_OCT_L_elastic -> 10536546_74y_OCT_L
      3. Remove modality tokens: 10536546_74y_OCT_L -> 10536546_74y_L
         (Normalize OCT's 10536546_74y_OCT and OCTA's 10536546_74y_OCTA to the same ID.)
      4. Match the trailing _L / _R -> side

    Return (subject_id, side or None). Modality tokens are removed for cross-modality alignment.
    """
    stem = path.stem

    # 1. Remove augmentation prefixes
    if "-" in stem:
        prefix, maybe_base = stem.split("-", 1)
        if prefix in _AUG_PREFIXES:
            stem = maybe_base

    # 2. Remove augmentation suffixes
    for suf in _AUG_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    # 3. Remove modality tokens
    stem = _MOD_TOKEN_RE.sub("", stem)

    # 4. Match the eye side
    m = _SIDE_SUFFIX_RE.match(stem)
    if m:
        return m.group("base"), m.group("side").upper()
    return stem, None


def list_images(class_dir: Path, extensions: Tuple[str, ...], recursive: bool) -> List[Path]:
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder does not exist: {class_dir}")

    iterator = class_dir.rglob("*") if recursive else class_dir.glob("*")
    files = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda p: p.name)


def group_lr_pairs(files: Sequence[Path]) -> List[Dict[str, object]]:
    """
    Group files in one class directory by subject_id, combining L/R originals and augmentations.

    Return a list with entries of the following form:
    {
        "subject_id": str,
        "files": [Path, ...],          # All subject views: original/augmented and left/right
        "paired": bool,
        "side_info": "L+R" / "L_only" / "R_only" / "single"
    }
    """
    buckets: Dict[str, Dict[str, List[Path]]] = defaultdict(
        lambda: {"L": [], "R": [], "X": []}
    )

    for p in files:
        subject_id, side = split_subject_side(p)
        if side in {"L", "R"}:
            buckets[subject_id][side].append(p)
        else:
            buckets[subject_id]["X"].append(p)

    groups: List[Dict[str, object]] = []

    for subject_id in sorted(buckets.keys()):
        b = buckets[subject_id]
        files_all = b["L"] + b["R"] + b["X"]
        if not files_all:
            continue

        has_l, has_r = bool(b["L"]), bool(b["R"])
        if has_l and has_r:
            side_info, paired = "L+R", True
        elif has_l:
            side_info, paired = "L_only", False
        elif has_r:
            side_info, paired = "R_only", False
        else:
            side_info, paired = "single", False

        groups.append({
            "subject_id": subject_id,
            "files": files_all,
            "paired": paired,
            "side_info": side_info,
        })

    return groups


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
    """
    Returns:
    {
        subject_id: {
            "label_idx": int,
            "label_name": str,
            "score": np.ndarray,   # Subject-level modality logits [num_classes]
            "files": str,
            "paired": bool,
            "side_info": str,
        }
    }
    """
    records: Dict[str, Dict[str, object]] = {}

    for label_idx, label_name in enumerate(class_order):
        class_dir = test_root / label_name
        files = list_images(class_dir, extensions, recursive)
        groups = group_lr_pairs(files)

        paired_count = sum(bool(g["paired"]) for g in groups)
        print(
            f"[{modality_name}] {label_name}: "
            f"{len(files)} files -> {len(groups)} samples, {paired_count} paired L/R samples"
        )

        for g in groups:
            subject_id = str(g["subject_id"])
            group_files = list(g["files"])

            if subject_id in records:
                old_label = records[subject_id]["label_name"]
                print(
                    f"[WARN] {modality_name} subject_id duplicated across classes: "
                    f"{subject_id}, old={old_label}, new={label_name}. New one is skipped."
                )
                continue

            score = predict_file_group(
                model=model,
                files=group_files,
                img_size=img_size,
                device=device,
            )

            records[subject_id] = {
                "label_idx": label_idx,
                "label_name": label_name,
                "score": score,
                "files": ";".join(str(p) for p in group_files),
                "paired": bool(g["paired"]),
                "side_info": str(g["side_info"]),
            }

    return records


def fuse_scores_entropy(
    oct_logits: np.ndarray,
    octa_logits: np.ndarray,
) -> np.ndarray:
    """
    Entropy-based eye-modality feature fusion (following cross_fusion.cross_feature_predict_entropy;
    use only OCT / OCTA features, excluding feature_bt / feature_sy and other features).

    - Apply row_normalize_l2 to each modality's logits (softmax then L2 if the row sum is not 1).
    - Compute per-sample modality entropies and obtain weights with softmax(-entropy):
      Lower-entropy modalities receive higher weights.
    - Sum weighted features into a fused distribution and use argmax for prediction.
    """
    f_oct = row_normalize_l2(oct_logits)
    f_octa = row_normalize_l2(octa_logits)

    e_oct = calculate_entropy(f_oct)
    e_octa = calculate_entropy(f_octa)

    entropies = np.stack([-e_oct, -e_octa], axis=1)  # [n_samples, 2]
    weights = softmax_np(entropies, axis=1)           # [n_samples, 2]

    return f_oct * weights[:, 0:1] + f_octa * weights[:, 1:2]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, score, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def compute_metrics(name: str, y_true: np.ndarray, score: np.ndarray) -> Dict[str, object]:
    y_pred = score.argmax(axis=1)
    return {
        "method": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": safe_auc(y_true, score),
    }


def main() -> None:
    args = parse_args()

    if len(args.class_order) != args.num_classes:
        raise ValueError(
            f"--class_order length ({len(args.class_order)}) must equal "
            f"--num_classes ({args.num_classes})."
        )

    if args.output_dir.exists() and not args.overwrite:
        # Do not delete the directory; guard against accidental overwriting of important files.
        existing = [args.output_dir / "predictions.csv", args.output_dir / "metrics.csv"]
        if any(p.exists() for p in existing):
            raise FileExistsError(f"{args.output_dir} already contains output files. Use --overwrite.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extensions = normalize_extensions(args.extensions)

    oct_device = get_device(args.oct_device)
    octa_device = get_device(args.octa_device)

    print("[INFO] Class order:", list(args.class_order))
    print("[INFO] fusion: entropy-weighted (eyeball OCT/OCTA logits only)")

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

    oct_ids = set(oct_records.keys())
    octa_ids = set(octa_records.keys())
    common_ids = sorted(oct_ids & octa_ids)
    oct_only_ids = sorted(oct_ids - octa_ids)
    octa_only_ids = sorted(octa_ids - oct_ids)

    # Relaxed policy: not all four images are required.
    # - Both modalities available -> entropy fusion;
    # - One modality available -> use its prediction without fusion; retain it in predictions and modality metrics;
    # - Only label conflicts (the same subject assigned different age groups across modalities) are recorded as skipped.
    skipped_rows = []
    rows = []

    matched_oct_logits, matched_octa_logits, matched_y = [], [], []
    oct_only_oct_logits, oct_only_y = [], []
    octa_only_octa_logits, octa_only_y = [], []

    for sid in common_ids:
        r_oct = oct_records[sid]
        r_octa = octa_records[sid]

        if int(r_oct["label_idx"]) != int(r_octa["label_idx"]):
            skipped_rows.append({
                "subject_id": sid,
                "reason": f"label_mismatch: OCT={r_oct['label_name']}, OCTA={r_octa['label_name']}",
            })
            continue

        matched_oct_logits.append(np.asarray(r_oct["score"], dtype=np.float64))
        matched_octa_logits.append(np.asarray(r_octa["score"], dtype=np.float64))
        matched_y.append(int(r_oct["label_idx"]))

        rows.append({
            "subject_id": sid,
            "fusion_type": "entropy",
            "true_label_idx": int(r_oct["label_idx"]),
            "true_label_name": str(r_oct["label_name"]),
            "oct_files": str(r_oct["files"]),
            "oct_paired": bool(r_oct["paired"]),
            "oct_side_info": str(r_oct["side_info"]),
            "octa_files": str(r_octa["files"]),
            "octa_paired": bool(r_octa["paired"]),
            "octa_side_info": str(r_octa["side_info"]),
        })

    for sid in oct_only_ids:
        r = oct_records[sid]
        oct_only_oct_logits.append(np.asarray(r["score"], dtype=np.float64))
        oct_only_y.append(int(r["label_idx"]))
        rows.append({
            "subject_id": sid,
            "fusion_type": "OCT_only",
            "true_label_idx": int(r["label_idx"]),
            "true_label_name": str(r["label_name"]),
            "oct_files": str(r["files"]),
            "oct_paired": bool(r["paired"]),
            "oct_side_info": str(r["side_info"]),
            "octa_files": "",
            "octa_paired": False,
            "octa_side_info": "missing",
        })

    for sid in octa_only_ids:
        r = octa_records[sid]
        octa_only_octa_logits.append(np.asarray(r["score"], dtype=np.float64))
        octa_only_y.append(int(r["label_idx"]))
        rows.append({
            "subject_id": sid,
            "fusion_type": "OCTA_only",
            "true_label_idx": int(r["label_idx"]),
            "true_label_name": str(r["label_name"]),
            "oct_files": "",
            "oct_paired": False,
            "oct_side_info": "missing",
            "octa_files": str(r["files"]),
            "octa_paired": bool(r["paired"]),
            "octa_side_info": str(r["side_info"]),
        })

    n_matched = len(matched_oct_logits)
    n_oct_only = len(oct_only_oct_logits)
    n_octa_only = len(octa_only_octa_logits)

    if n_matched == 0 and n_oct_only == 0 and n_octa_only == 0:
        raise RuntimeError("No OCT or OCTA samples found. Please check file names and class folders.")

    C = args.num_classes

    # Matched subjects: entropy fusion and per-modality softmax
    if n_matched > 0:
        m_oct_l = np.vstack(matched_oct_logits)
        m_octa_l = np.vstack(matched_octa_logits)
        m_oct_prob = softmax_np(m_oct_l, axis=1)
        m_octa_prob = softmax_np(m_octa_l, axis=1)
        m_fused = fuse_scores_entropy(m_oct_l, m_octa_l)
        m_y = np.asarray(matched_y, dtype=int)
    else:
        m_oct_prob = np.zeros((0, C)); m_octa_prob = np.zeros((0, C))
        m_fused = np.zeros((0, C)); m_y = np.zeros((0,), dtype=int)

    # Single modality: softmax
    o_oct_prob = (softmax_np(np.vstack(oct_only_oct_logits), axis=1)
                  if n_oct_only > 0 else np.zeros((0, C)))
    o_y = np.asarray(oct_only_y, dtype=int) if n_oct_only > 0 else np.zeros((0,), dtype=int)
    a_octa_prob = (softmax_np(np.vstack(octa_only_octa_logits), axis=1)
                   if n_octa_only > 0 else np.zeros((0, C)))
    a_y = np.asarray(octa_only_y, dtype=int) if n_octa_only > 0 else np.zeros((0,), dtype=int)

    # Align prediction rows: [matched | oct_only | octa_only]
    N = n_matched + n_oct_only + n_octa_only
    oct_scores_full = np.full((N, C), np.nan)
    octa_scores_full = np.full((N, C), np.nan)
    fusion_scores_full = np.full((N, C), np.nan)

    oct_scores_full[:n_matched] = m_oct_prob
    oct_scores_full[n_matched:n_matched + n_oct_only] = o_oct_prob
    # Keep OCT values as NaN for octa_only rows

    octa_scores_full[:n_matched] = m_octa_prob
    octa_scores_full[n_matched + n_oct_only:] = a_octa_prob
    # Keep OCTA values as NaN for oct_only rows

    fusion_scores_full[:n_matched] = m_fused
    fusion_scores_full[n_matched:n_matched + n_oct_only] = o_oct_prob    # Fall back to OCT
    fusion_scores_full[n_matched + n_oct_only:] = a_octa_prob           # Fall back to OCTA

    def _argmax_rows(arr: np.ndarray) -> np.ndarray:
        out = np.full(arr.shape[0], -1, dtype=int)
        for i in range(arr.shape[0]):
            if not np.all(np.isnan(arr[i])):
                out[i] = int(np.nanargmax(arr[i]))
        return out

    def _name(idx: int) -> str:
        return args.class_order[idx] if idx >= 0 else ""

    oct_pred = _argmax_rows(oct_scores_full)
    octa_pred = _argmax_rows(octa_scores_full)
    fused_pred = _argmax_rows(fusion_scores_full)

    pred_df = pd.DataFrame(rows)
    pred_df["oct_pred_idx"] = oct_pred
    pred_df["oct_pred_name"] = [_name(i) for i in oct_pred]
    pred_df["octa_pred_idx"] = octa_pred
    pred_df["octa_pred_name"] = [_name(i) for i in octa_pred]
    pred_df["fusion_pred_idx"] = fused_pred
    pred_df["fusion_pred_name"] = [_name(i) for i in fused_pred]

    for i, cls in enumerate(args.class_order):
        safe_cls = str(cls).replace("+", "plus").replace("/", "_")
        pred_df[f"oct_score_{safe_cls}"] = oct_scores_full[:, i]
        pred_df[f"octa_score_{safe_cls}"] = octa_scores_full[:, i]
        pred_df[f"fusion_score_{safe_cls}"] = fusion_scores_full[:, i]

    # Metrics: OCT includes matched + oct_only subjects; likewise for OCTA; fusion uses matched subjects only.
    metrics = []
    if n_matched + n_oct_only > 0:
        metrics.append(compute_metrics(
            "OCT", np.concatenate([m_y, o_y]), np.vstack([m_oct_prob, o_oct_prob])))
    if n_matched + n_octa_only > 0:
        metrics.append(compute_metrics(
            "OCTA", np.concatenate([m_y, a_y]), np.vstack([m_octa_prob, a_octa_prob])))
    if n_matched > 0:
        metrics.append(compute_metrics("OCT+OCTA(entropy)", m_y, m_fused))
    metrics_df = pd.DataFrame(metrics)

    pred_path = args.output_dir / "predictions.csv"
    metrics_path = args.output_dir / "metrics.csv"
    skipped_path = args.output_dir / "skipped_samples.csv"
    report_path = args.output_dir / "fusion_report.xlsx"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        pred_df.to_excel(writer, sheet_name="predictions", index=False)
        pd.DataFrame(skipped_rows).to_excel(writer, sheet_name="skipped", index=False)

    print("\nDone.")
    print(f"Samples in predictions: {len(pred_df)} "
          f"(entropy fusion={n_matched}, OCT_only={n_oct_only}, OCTA_only={n_octa_only})")
    print(f"Skipped (label mismatch only): {len(skipped_rows)}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
