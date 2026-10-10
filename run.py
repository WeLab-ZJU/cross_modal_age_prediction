#!/usr/bin/env python3
"""Dispatch each workflow in a fresh process without mixing local imports."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {
    "human": {
        "train-eye": "train_eye.py",
        "evaluate-eye": "train_eye.py",
        "fuse-eye": "fuse_oct_octa_human.py",
        "fuse-eye-age": "fuse_oct_octa_human2.py",
        "visualize": "visualize_all_best_models.py",
    },
    "mouse": {
        "train-eye": "train_eye.py",
        "evaluate-eye": "train_eye.py",
        "train-gait": "train_gait.py",
        "train-behavior": "train_behavior.py",
        "fuse": "fuse_modalities.py",
        "continuous-age": "predict_continuous_age.py",
        "generate-demo": "_generate_demo_data.py",
    },
}


def build_command(species: str, workflow: str, arguments: list[str]) -> list[str]:
    script = ROOT / species / ENTRYPOINTS[species][workflow]
    forwarded = list(arguments)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    if workflow == "evaluate-eye" and "--eval" not in forwarded:
        forwarded.insert(0, "--eval")
    return [sys.executable, str(script), *forwarded]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified human/mouse workflow launcher.",
        epilog="Paths passed to workflows are relative to the repository root. "
               "Use absolute paths for external data. --dry-run precedes the species.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without importing ML dependencies.")
    species_parsers = parser.add_subparsers(dest="species", required=True)
    for species, workflows in ENTRYPOINTS.items():
        sub = species_parsers.add_parser(species)
        sub.add_argument("workflow", choices=workflows)
        sub.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = build_command(args.species, args.workflow, args.arguments)
    if args.dry_run:
        print(f"Working directory: {ROOT}")
        print(shlex.join(command))
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
