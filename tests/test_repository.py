"""Launcher and source checks; no ML libraries, data, network or GPU required."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_launcher", ROOT / "run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RepositoryTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            if not any(part.startswith(".") for part in path.relative_to(ROOT).parts):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    compile(path.read_bytes(), str(path), "exec")

    def test_shell_syntax(self):
        for path in ROOT.rglob("*.sh"):
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoints_exist_and_preserve_arguments(self):
        for species, workflows in launcher.ENTRYPOINTS.items():
            for workflow, filename in workflows.items():
                command = launcher.build_command(species, workflow, ["--", "--help"])
                self.assertEqual(command[:2], [sys.executable, str(ROOT / species / filename)])
                self.assertTrue(Path(command[1]).is_file())
                self.assertEqual(command[-1], "--help")
                self.assertEqual(command.count("--eval"), int(workflow == "evaluate-eye"))

    def test_evaluate_does_not_duplicate_flag(self):
        command = launcher.build_command("human", "evaluate-eye", ["--eval", "--batch_size", "1"])
        self.assertEqual(command.count("--eval"), 1)

    def test_dry_run_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "run.py"), "--dry-run",
                 "mouse", "evaluate-eye", "--resume", "path with spaces/model.pth"],
                cwd=temp, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT), result.stdout)
        self.assertIn("--eval", result.stdout)
        self.assertIn("'path with spaces/model.pth'", result.stdout)

    def test_training_help_enabled(self):
        for species in ("human", "mouse"):
            tree = ast.parse((ROOT / species / "train_eye.py").read_text())
            self.assertTrue(any(
                isinstance(node, ast.keyword) and node.arg == "add_help"
                and isinstance(node.value, ast.Constant) and node.value.value is True
                for node in ast.walk(tree)
            ))

    def test_local_imports_present(self):
        for species in ("human", "mouse"):
            for filename in ("engine_finetune.py", "models_vit.py", "util/__init__.py",
                             "util/datasets.py", "util/misc.py", "util/lr_decay.py",
                             "util/lr_sched.py", "util/pos_embed.py"):
                self.assertTrue((ROOT / species / filename).is_file(), filename)

    def test_no_machine_paths_in_source_text(self):
        # This test itself expresses the prefixes in separate tokens.
        forbidden = ("/mnt/" + "sdb1/", "/Users/" + "wyh/", "/home/" + "amax/")
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".txt", ".md", ".json"}:
                text = path.read_text()
                self.assertFalse(any(prefix in text for prefix in forbidden), str(path))

    def test_new_shell_flags_match_target_parsers(self):
        mappings = {
            "scripts/train_eye.sh": "human/train_eye.py",
            "scripts/evaluate_eye.sh": "human/train_eye.py",
            "scripts/fuse_human_eye.sh": "human/fuse_oct_octa_human2.py",
            "mouse/demo/run_demo_gait.sh": "mouse/train_gait.py",
            "mouse/demo/run_demo_behavior.sh": "mouse/train_behavior.py",
            "mouse/demo/run_demo_continuous_age.sh": "mouse/predict_continuous_age.py",
        }
        for shell, script in mappings.items():
            tree = ast.parse((ROOT / script).read_text())
            flags = {
                value.value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                for value in node.args
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            }
            used = set(re.findall(r"--[a-z][a-z_]+", (ROOT / shell).read_text()))
            used -= {"--nproc_per_node", "--master_port"}
            self.assertFalse(used - flags, f"{shell}: unsupported flags {used - flags}")

    def test_shell_required_paths_fail_before_launch(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"DATA_ROOT", "HCF_CKPT", "CHECKPOINT",
                                      "OCT_CKPT", "OCTA_CKPT", "SPECIES"}}
        for shell in ("train_eye.sh", "evaluate_eye.sh", "fuse_human_eye.sh"):
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / shell)], env=environment,
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Set ", result.stderr)


if __name__ == "__main__":
    unittest.main()
