---
name: root-gnn-dgl-inference
description: Use when the user asks to score ROOT files, add GNN score branches, launch MPI inference, or verify inference outputs in root_gnn_dgl, including scripts/inference.py, jobs/inference/run_inference.py, ROOT branch checks, and NPZ or ROOT output audits.
---

# root-gnn-dgl-inference

Use this skill to score ntuples with trained models and verify the outputs.

## Environment

- Run inference in an environment with PyROOT available.
- The repo README says the conda environment is required for inference because the podman training image does not include ROOT.

## Entry point

Run from the repo root:

```bash
python scripts/inference.py \
  --target <input.root> \
  --destination <output.root> \
  --config <config.yaml> \
  --branch_name <branch_name> \
  --chunks 1 \
  --chunkno 0 \
  --write
```

## Multi-model inference

The script accepts multiple configs and multiple branch names in one run:

```bash
python scripts/inference.py \
  --target <input.root> \
  --destination <output.root> \
  --config config_a.yaml config_b.yaml \
  --branch_name score_a score_b \
  --chunks 1 \
  --chunkno 0 \
  --write
```

The number of configs and branch names must match.

## Checkpoint selection

- With the default `--ckpt -1`, the script selects the best epoch from `training.log` using `--var` and `--mode`.
- Use `--ckpt <n>` to force a specific checkpoint.
- If `--destination` is omitted, the script writes under `<Training_Directory>/inference/`.

## Output modes

- `--write` creates a new ROOT file and adds score branches.
- Without `--write`, the script saves an `.npz` bundle containing scores, labels, and tracking info.
- Use `--clobber` when reusing an existing destination path.

## Parallel inference

Use the generic MPI/local wrapper for many files or many models:

```bash
mpirun -np <num_ranks> python jobs/inference/run_inference.py \
  --sample-config <config-with-target-datasets.yaml> \
  --config-dir <directory-of-model-configs> \
  --output-dir <output-directory> \
  --write
```

Useful variants:

```bash
python jobs/inference/run_inference.py --target '<glob-or-file.root>' --config model.yaml --output-dir scores --write
python jobs/inference/run_inference.py --sample-config samples.yaml --config model_a.yaml model_b.yaml --branch-name score_a score_b --output-dir scores --write
python jobs/inference/run_inference.py --sample-config samples.yaml --config-dir configs/run_3_ttH --output-dir scores --write --test
```

Wrapper behavior:

- `--sample-config` discovers target ROOT files from every dataset's `args.raw_dir` and `args.file_names`.
- `--target` accepts explicit files or glob patterns.
- `--config-dir` discovers model configs from `*.yaml`; `--config` accepts explicit config files.
- Branch names default to each model config's `Training_Name` plus `_score`.
- `--branch-name` may override branch names, but the count must match the config count.
- `--test` prints the planned `scripts/inference.py` commands without running them.
- With MPI, tasks are split by target file across ranks; without MPI it runs serially.
- GPU assignment is local rank modulo 4 through `CUDA_VISIBLE_DEVICES`.

## Repo-specific behavior

- The first config's first dataset is used as the template dataset. The script rewrites `raw_dir`, `file_names`, `save_dir`, `chunks`, `process_chunks`, and optionally `tree_name` at runtime.
- Pass `--tree <name>` if the ROOT tree name differs from the config default.
- Chunked inference writes per-chunk outputs; merging those outputs is a separate step.
- Job wrappers should derive the repo root from their own path; avoid adding hard-coded checkout paths.

## Audit the outputs

Start with basic runtime evidence if you have a log:

- `Writing to file`
- `Input entries:`
- `Output entries:`
- `Wrote scores to`
- absence of `Traceback`

For ROOT outputs, prefer `uproot`:

```bash
python - <<'PY'
import numpy as np
import uproot
path = "<output.root>"
branches = ["<score_branch>"]
tree = uproot.open(path)["output"]
print("entries", tree.num_entries)
for branch in branches:
    arr = tree[branch].array(library="np")
    print(branch, len(arr), np.isnan(arr).sum(), float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)))
PY
```

Treat inference as valid only if:

- the destination file exists
- every requested score branch exists
- output entry count matches the input tree
- score arrays contain no NaNs
- score arrays are not constant

For multi-model inference, every branch must exist and branch statistics should usually differ unless the models are intentionally identical.

Without `--write`, inspect the `.npz` keys `scores`, `labels`, and `tracking_info` and verify array lengths and NaN counts.
