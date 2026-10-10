#!/usr/bin/env python3
"""Visualize every best checkpoint against its matching ImageFolder dataset."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retf-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from pytorch_grad_cam import EigenCAM, GradCAM
except ImportError as exc:
    raise RuntimeError(
        "pytorch-grad-cam is required; run this program in the retf environment."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import models_vit


METHOD_MAP = {
    "gradcam": GradCAM,
    "eigencam": EigenCAM,
}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_METHODS = ("gradcam", "eigencam")
DEFAULT_LAYER_SUFFIXES = (
    "attn",
    "drop_path1",
    "drop_path2",
    "norm2",
    "mlp",
)


@dataclass(frozen=True)
class CheckpointRecord:
    group: str
    modality: str
    checkpoint: Path
    data_root: Path
    output_base: Path


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    img_size: int
    patch_size: int
    num_classes: int
    drop_path_rate: float
    global_pool: bool


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_checkpoint_value(checkpoint_args: Any, name: str, default: Any) -> Any:
    if checkpoint_args is None:
        return default
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def infer_arch(checkpoint: Path, checkpoint_args: Any, model_folder: str) -> str:
    arch = get_checkpoint_value(checkpoint_args, "model", None)
    if arch in models_vit.__dict__:
        return arch

    candidate = model_folder.removeprefix("mae-")
    if candidate in models_vit.__dict__:
        return candidate

    match = re.search(r"(vit_(?:base|large|huge)_patch\d+)", checkpoint.as_posix())
    if match and match.group(1) in models_vit.__dict__:
        return match.group(1)
    raise ValueError(f"Cannot infer a supported ViT architecture from {checkpoint}")


def discover_checkpoints(
    outputs_roots: Sequence[Path],
    datasets_root: Path,
    output_root: Path,
) -> list[CheckpointRecord]:
    records: list[CheckpointRecord] = []
    for outputs_root_item in outputs_roots:
        outputs_root_item = outputs_root_item.resolve()
        group = outputs_root_item.name
        if not outputs_root_item.is_dir():
            print(f"[WARN] output root does not exist: {outputs_root_item}")
            continue

        candidates = sorted(
            path
            for path in outputs_root_item.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".pth", ".pt"}
            and "best" in path.name.lower()
        )
        for checkpoint in candidates:
            relative = checkpoint.relative_to(outputs_root_item)
            if len(relative.parts) < 2:
                print(f"[WARN] cannot derive modality from {checkpoint}")
                continue

            modality = relative.parts[0]
            data_root = datasets_root / group / modality
            if not data_root.is_dir():
                print(
                    f"[WARN] matching dataset is missing for {checkpoint}: {data_root}"
                )
                continue

            records.append(
                CheckpointRecord(
                    group=group,
                    modality=modality,
                    checkpoint=checkpoint,
                    data_root=data_root,
                    output_base=output_root / group / relative.parent,
                )
            )

    records.sort(key=lambda item: item.checkpoint.as_posix())
    return records


def load_model(
    record: CheckpointRecord,
) -> tuple[torch.nn.Module, ModelConfig, dict[str, Any]]:
    checkpoint_data = torch.load(record.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict):
        raise ValueError(f"Unsupported checkpoint format: {record.checkpoint}")

    checkpoint_args = checkpoint_data.get("args")
    model_folder = record.checkpoint.relative_to(record.checkpoint.parents[3]).parts[0]
    arch = infer_arch(record.checkpoint, checkpoint_args, model_folder)

    img_size = int(
        get_checkpoint_value(
            checkpoint_args,
            "input_size",
            next(
                (
                    int(match.group(1))
                    for match in re.finditer(r"eye-(\d+)", record.checkpoint.as_posix())
                ),
                224,
            ),
        )
    )
    num_classes = int(get_checkpoint_value(checkpoint_args, "nb_classes", 4))
    drop_path_rate = float(get_checkpoint_value(checkpoint_args, "drop_path", 0.1))
    global_pool = bool(get_checkpoint_value(checkpoint_args, "global_pool", True))

    model_factory = models_vit.__dict__[arch]
    model = model_factory(
        img_size=img_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    state_dict = checkpoint_data.get("model")
    if state_dict is None:
        state_dict = checkpoint_data.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint_data
    if not isinstance(state_dict, dict):
        raise ValueError(f"No state dict found in {record.checkpoint}")

    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    load_message = model.load_state_dict(cleaned_state_dict, strict=False)
    if load_message.missing_keys or load_message.unexpected_keys:
        print(
            f"[WARN] checkpoint load mismatch for {record.checkpoint}: "
            f"missing={load_message.missing_keys}, "
            f"unexpected={load_message.unexpected_keys}"
        )

    patch_size = getattr(model.patch_embed, "patch_size", 16)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    config = ModelConfig(
        arch=arch,
        img_size=img_size,
        patch_size=int(patch_size),
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )
    return model, config, {
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "checkpoint_args": str(checkpoint_args),
    }


def resolve_layer(layer_path: str, model: torch.nn.Module) -> torch.nn.Module:
    current: Any = model
    for component in layer_path.strip().split("."):
        index_match = re.fullmatch(r"([^\[]+)((?:\[\d+\])*)", component)
        if index_match is None:
            raise AttributeError(f"Invalid layer component: {component}")
        name, indices = index_match.groups()
        if name:
            if name.isdigit():
                current = current[int(name)]
            else:
                current = getattr(current, name)
        for index in re.findall(r"\[(\d+)\]", indices):
            current = current[int(index)]
    return current


def default_target_layers(model: torch.nn.Module) -> list[str]:
    last_block = len(model.blocks) - 1
    return [
        f"blocks.{last_block}.{suffix}" for suffix in DEFAULT_LAYER_SUFFIXES
    ]


def build_reshape_transform() -> Any:
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor[:, 1:, :]
        token_count = tensor.shape[1]
        grid_size = int(math.sqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(
                f"Expected a square patch grid, got {token_count} patch tokens"
            )
        tensor = tensor.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
        return tensor.permute(0, 3, 1, 2)

    return reshape_transform


def make_cam(
    method: str,
    model: torch.nn.Module,
    target_layer: str,
) -> Any:
    cam_class = METHOD_MAP[method]
    return cam_class(
        model=model,
        target_layers=[resolve_layer(target_layer, model)],
        reshape_transform=build_reshape_transform(),
    )


def image_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def output_path_for(
    output_base: Path,
    method: str,
    target_layer: str,
    data_root: Path,
    image_path: Path,
    output_format: str,
) -> Path:
    relative = image_path.relative_to(data_root)
    layer_folder = target_layer.replace(os.sep, "_").replace("/", "_")
    return (
        output_base
        / method
        / layer_folder
        / relative.parent
        / f"{relative.stem}_{method}.{output_format}"
    )


def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_batch(
    paths: Sequence[Path],
    transform: transforms.Compose,
) -> tuple[torch.Tensor, list[Image.Image], list[tuple[int, int]]]:
    tensors: list[torch.Tensor] = []
    originals: list[Image.Image] = []
    original_sizes: list[tuple[int, int]] = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            original = image.copy()
        originals.append(original)
        original_sizes.append(original.size)
        tensors.append(transform(image))
    return torch.stack(tensors), originals, original_sizes


def render_cam(
    original: Image.Image,
    original_size: tuple[int, int],
    grayscale_cam: np.ndarray,
    destination: Path,
    output_format: str,
) -> None:
    grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
    grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)
    cam_image = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    cam_image = cam_image.resize(original_size, resampling)
    resized_cam = np.asarray(cam_image, dtype=np.float32) / 255.0

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(original)
    attention = axis.imshow(resized_cam, cmap="jet", alpha=0.8)
    axis.axis("off")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    colorbar_axis = make_axes_locatable(axis).append_axes(
        "right", size="5%", pad=0.05
    )
    figure.colorbar(attention, cax=colorbar_axis)
    figure.tight_layout()
    figure.savefig(
        destination,
        bbox_inches="tight",
        pad_inches=0,
        format=output_format,
        transparent=True,
    )
    plt.close(figure)


def append_failure(output_root: Path, image_path: Path, error: Exception) -> None:
    failure_path = output_root / "visualization_failures.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "image": str(image_path),
        "error": repr(error),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    with failure_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_method_layer(
    model: torch.nn.Module,
    device: torch.device,
    record: CheckpointRecord,
    config: ModelConfig,
    split: str,
    method: str,
    target_layer: str,
    output_format: str,
    batch_size: int,
    overwrite: bool,
    limit_per_split: int | None,
    output_root: Path,
) -> tuple[int, int, int]:
    split_root = record.data_root / split
    if not split_root.is_dir():
        print(f"[WARN] split does not exist: {split_root}")
        return 0, 0, 0

    paths = image_files(split_root)
    if limit_per_split is not None:
        paths = paths[:limit_per_split]

    work_items: list[tuple[Path, Path]] = []
    for image_path in paths:
        destination = output_path_for(
            record.output_base,
            method,
            target_layer,
            record.data_root,
            image_path,
            output_format,
        )
        if overwrite or not destination.is_file() or destination.stat().st_size == 0:
            work_items.append((image_path, destination))

    skipped = len(paths) - len(work_items)
    if not work_items:
        print(
            f"[SKIP] {record.group}/{record.modality} {split} "
            f"{method}/{target_layer}: {len(paths)} existing files"
        )
        return 0, skipped, 0

    transform = make_transform(config.img_size)
    processed = 0
    failed = 0
    cam = make_cam(method, model, target_layer)
    progress = tqdm(
        range(0, len(work_items), batch_size),
        desc=f"{record.group}/{record.modality}/{split} {method}/{target_layer}",
        unit="batch",
    )
    try:
        for start in progress:
            batch_items = work_items[start : start + batch_size]
            batch_paths = [item[0] for item in batch_items]
            destinations = [item[1] for item in batch_items]
            originals: list[Image.Image] = []
            try:
                input_tensor, originals, original_sizes = load_batch(
                    batch_paths, transform
                )
                input_tensor = input_tensor.to(device)
                grayscale_cams = cam(input_tensor=input_tensor)
                for original, original_size, grayscale_cam, destination in zip(
                    originals, original_sizes, grayscale_cams, destinations
                ):
                    render_cam(
                        original,
                        original_size,
                        grayscale_cam,
                        destination,
                        output_format,
                    )
                    processed += 1
                progress.set_postfix(done=processed, failed=failed)
            except Exception as batch_error:
                if len(batch_items) == 1:
                    failed += 1
                    append_failure(output_root, batch_paths[0], batch_error)
                    print(f"[ERROR] {batch_paths[0]}: {batch_error}")
                    continue

                for image_path, destination in batch_items:
                    try:
                        input_tensor, originals, original_sizes = load_batch(
                            [image_path], transform
                        )
                        grayscale_cam = cam(input_tensor=input_tensor.to(device))[0]
                        render_cam(
                            originals[0],
                            original_sizes[0],
                            grayscale_cam,
                            destination,
                            output_format,
                        )
                        processed += 1
                    except Exception as image_error:
                        failed += 1
                        append_failure(output_root, image_path, image_error)
                        print(f"[ERROR] {image_path}: {image_error}")
                progress.set_postfix(done=processed, failed=failed)
    finally:
        if hasattr(cam, "clear_hooks"):
            cam.clear_hooks()
        del cam
        for original in originals:
            original.close()

    return processed, skipped, failed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[WARN] requested device {requested}, but CUDA is unavailable; "
            "falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(requested)


def write_manifest(
    output_root: Path,
    records: Sequence[CheckpointRecord],
    methods: Sequence[str],
    target_layers: str,
    splits: Sequence[str],
    device: torch.device,
    output_format: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "methods": list(methods),
        "target_layers": target_layers,
        "splits": list(splits),
        "output_format": output_format,
        "checkpoints": [
            {
                "group": record.group,
                "modality": record.modality,
                "checkpoint": str(record.checkpoint),
                "data_root": str(record.data_root),
                "output_base": str(record.output_base),
            }
            for record in records
        ],
    }
    with (output_root / "visualization_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAM visualization for all best ViT checkpoints."
    )
    parser.add_argument(
        "--outputs-roots",
        nargs="+",
        default=[
            "outputs/human/dataset",
            "outputs/human/merge",
        ],
    )
    parser.add_argument(
        "--datasets-root",
        default="data/human",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/human/visualization",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--target-layers",
        default="auto",
        help="Comma-separated layer paths, or auto for the final block's five layers.",
    )
    parser.add_argument("--splits", default="train,test,val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--output-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        raise ValueError("--limit-per-split must be at least 1")

    outputs_roots = [Path(path).resolve() for path in args.outputs_roots]
    datasets_root = Path(args.datasets_root).resolve()
    output_root = Path(args.output_root).resolve()
    methods = parse_csv(args.methods.lower())
    splits = parse_csv(args.splits)
    unknown_methods = [method for method in methods if method not in METHOD_MAP]
    if unknown_methods:
        raise ValueError(
            f"Unknown method(s): {unknown_methods}; supported: {sorted(METHOD_MAP)}"
        )
    if not methods:
        raise ValueError("At least one CAM method is required")
    if not splits:
        raise ValueError("At least one split is required")

    device = resolve_device(args.device)
    records = discover_checkpoints(outputs_roots, datasets_root, output_root)
    if not records:
        raise FileNotFoundError("No best checkpoint was found under the output roots")

    print(f"Found {len(records)} best checkpoints.")
    for record in records:
        print(f"  {record.group}/{record.modality}: {record.checkpoint}")

    if args.dry_run:
        for record in records:
            for split in splits:
                count = len(image_files(record.data_root / split))
                print(f"  {record.group}/{record.modality}/{split}: {count} images")
        return

    write_manifest(
        output_root,
        records,
        methods,
        args.target_layers,
        splits,
        device,
        args.output_format,
    )

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    started = time.time()

    for index, record in enumerate(records, start=1):
        print(f"\n[{index}/{len(records)}] Loading {record.checkpoint}")
        model, config, checkpoint_info = load_model(record)
        print(f"Model config: {asdict(config)}")
        print(f"Checkpoint info: {checkpoint_info}")
        model.eval()
        model.to(device)
        target_layers = (
            default_target_layers(model)
            if args.target_layers.strip().lower() == "auto"
            else parse_csv(args.target_layers)
        )
        print(f"Target layers: {target_layers}")

        for split in splits:
            for method in methods:
                for target_layer in target_layers:
                    processed, skipped, failed = process_method_layer(
                        model=model,
                        device=device,
                        record=record,
                        config=config,
                        split=split,
                        method=method,
                        target_layer=target_layer,
                        output_format=args.output_format,
                        batch_size=args.batch_size,
                        overwrite=args.overwrite,
                        limit_per_split=args.limit_per_split,
                        output_root=output_root,
                    )
                    total_processed += processed
                    total_skipped += skipped
                    total_failed += failed

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - started
    print(
        f"\nCompleted in {elapsed / 3600:.2f} hours: "
        f"processed={total_processed}, skipped={total_skipped}, failed={total_failed}."
    )
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
