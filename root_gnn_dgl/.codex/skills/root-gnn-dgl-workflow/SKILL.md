---
name: root-gnn-dgl-workflow
description: Use when the user asks to run or review the full root_gnn_dgl workflow, such as run_demo.sh, a full pretraining-to-finetuning-to-inference campaign, or a stage-by-stage pass, warning, fail audit across environment setup, data preparation, training, inference, and outputs.
---

# root-gnn-dgl-workflow

Use this skill when the user wants an end-to-end workflow rather than a single isolated stage.

## Shipped demo

Run from the repo root:

```bash
source run_demo.sh
```

The demo does:

1. graph prep for multiclass pretraining
2. multiclass pretraining
3. graph prep for binary classification
4. from-scratch binary training
5. finetuned binary training
6. inference with two output score branches

## Before running the workflow

- Check GPU availability with `nvidia-smi` or request an interactive node with `jobs/interactive.sh`.
- Confirm the target data and output directories in `run_demo.sh` exist and are writable.
- Confirm `configs/stats_100K/finetuning_ttH_CP_even_vs_odd.yaml` points at the checkpoint you actually want to finetune from.

## Workflow audit order

When reviewing a campaign, check stages in this order:

1. environment readiness
2. data-prep outputs
3. training submission and queue state
4. training logs and checkpoints
5. inference outputs

Use the retained stage skills for each check:

- `root-gnn-dgl-env-setup`
- `root-gnn-dgl-data-preparation`
- `root-gnn-dgl-training`
- `root-gnn-dgl-inference`

## Output style

Return a short status for each stage:

- `pass`: evidence is consistent with a healthy stage
- `warning`: stage likely worked but still needs a follow-up check
- `fail`: concrete blocker or corrupted or missing artifact found

Repo-specific workflow blockers:

- pending jobs in `sqs` with `Priority` are waiting, not failed
- missing prebatched `.bin` files block `--preshuffle` training
- missing `pretraining_path` blocks finetuning
- ROOT outputs without the requested score branches are inference failures even if the file exists

## When to adapt instead of sourcing the demo

- If you only want one stage, call the underlying prep, training, or inference script directly.
- If dataset paths, branch names, or chunk counts differ, copy the command pattern from `run_demo.sh` and adjust the values instead of editing the demo in place.
