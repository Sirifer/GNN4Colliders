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

The repo includes an MPI wrapper:

```bash
mpirun -np <num_ranks> python jobs/inference/run_inference.py
```

That script is campaign-specific, with hard-coded file lists, configs, branches, and destinations. Patch or copy it before using it for a different campaign.

## Repo-specific behavior

- The first config's first dataset is used as the template dataset. The script rewrites `raw_dir`, `file_names`, `save_dir`, `chunks`, `process_chunks`, and optionally `tree_name` at runtime.
- Pass `--tree <name>` if the ROOT tree name differs from the config default.
- Chunked inference writes per-chunk outputs; merging those outputs is a separate step.
- The script prepends a hard-coded repo path near the top of the file. If imports fail outside the expected NERSC layout, fix that path first.

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
