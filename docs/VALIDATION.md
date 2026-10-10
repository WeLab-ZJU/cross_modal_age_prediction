# Packaging and English-Edition Validation

Date: October 10, 2026.

## Packaging checks

```bash
python -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s tests -v
```

The local interpreters are Python 3.14.6 and Python 3.9.6.
The packaging suite contains **11 standard-library tests**. Passing these
checks on Python 3.9 does not imply that the training workflows support
Python 3.9; use Python 3.10+ for the actual workflows.

Coverage:

1. Static compilation of all 31 Python files, without importing ML libraries.
2. `bash -n` syntax checks for all six current shell scripts.
3. Existence of every unified entry point and preservation of forwarded arguments.
4. Automatic, nonduplicated `--eval` insertion by `evaluate-eye`.
5. Dry-run execution from outside the repository, including arguments with spaces.
6. Enabled argparse help for the two primary eye-training entry points.
7. Presence of both species' models, training engines, and utility modules.
8. Agreement between packaged files and the source-manifest SHA-256 hashes.
9. Absence of original personal server paths in packaged text.
10. Agreement between new shell arguments and target Python parsers,
    with torch launcher arguments handled separately.
11. Failure before training starts when required data or checkpoint paths
    have not been provided.

## Translation-specific checks

The English edition is additionally checked for:

- The same relative file set as the consolidated source.
- No remaining Chinese text in UTF-8 files.
- Matching Python abstract syntax trees after ignoring docstrings and
  normalizing the single translated error message.
- Matching Python executable tokens after excluding comments and
  docstrings and normalizing that error message.
- Matching active shell files, configuration files, and dependency lists.
- Updated packaged-file hashes while retaining original-source hashes.
- Unchanged files in the neighboring consolidated source directory.

These checks verify translation and packaging, not scientific validity.

### English-edition results

All checks above passed on October 10, 2026:

- All 11 packaging tests passed on both local interpreters.
- Both repositories contain the same 79 relative file paths.
- No Chinese text remains in the English edition.
- All 31 Python files retain matching executable tokens and abstract
  syntax trees under the explanatory-text normalization described above.
- Current shell scripts, dependency lists, and `.gitignore` are unchanged.
- All 61 source mappings have valid packaged hashes in both editions,
  with the original-source hashes preserved.

No training or inference was performed as part of these checks.

## Source completeness

The original consolidation retained mappings for all 60 source `.py`,
`.sh`, and `.md` files. Supplying the missing mouse engine produces
61 source-to-destination mappings. No Finder files, bytecode caches, real
data, weights, training outputs, or `.git` directory are intended for
inclusion. No GitHub commit or push was performed.

## Not executed

- Installation or locking of a scientific-computing / deep-learning environment.
- Actual `--help` execution for training entry points requiring third-party libraries.
- ViT forward/backward passes, checkpoint loading, or CUDA/DDP execution.
- End-to-end execution of the three tabular demos.
- Real human/mouse training, fusion, age scoring, or Grad-CAM visualization.
- Performance reproduction, subject alignment checks, or real-data label validation.

The local environment lacks key dependencies including PyTorch, pandas,
scikit-learn, timm, and matplotlib. Static and translation checks
**do not demonstrate successful training or evaluation**.
Review `docs/MIGRATION.md` in the original experimental environment before
formal use.
