---
name: root-gnn-dgl-plotting
description: Use when the user asks to plot training curves, regenerate training.png, build the sweep PDF from plotting/training_performance.py, compare training runs, or extract Loss, Accuracy, Test_Loss, Test_AUC, and timing information from training.log files in root_gnn_dgl.
---

# root-gnn-dgl-plotting

Use this skill when the task is about plots or metrics derived from `training.log`.

## Single-run plot regeneration

The training script can regenerate the per-run PNG directly:

```bash
python scripts/training_script.py --config <config.yaml> --plot
```

That uses `root_gnn_base.utils.read_log()` and `root_gnn_base.utils.plot_log()` to rebuild `training.png` from `Training_Directory/training.log`.

`plot_log()` produces a 2x2 figure with:

- cumulative time in seconds
- train and test loss
- accuracy
- test AUC

Be aware that `plot_log()` fixes the accuracy axis to `(0.44, 0.56)`, which may be too narrow for some runs.

## Sweep-level plotting

Use the dedicated plotting script when the user wants a PDF comparing shipped sweeps:

```bash
python plotting/training_performance.py
python plotting/training_performance.py --output <output.pdf>
```

The script currently plots two config groups:

- `pretraining`
- `higgs_production`

It writes one PDF page per group and resolves each run's `Training_Directory` from its config.

## What the plotting script reads from training.log

`plotting/training_performance.py` parses rows that start with `Epoch` and extracts:

- `Epoch`
- `Loss`
- `Accuracy`
- `Test_Loss`
- `Test_AUC`
- `Time`

It also computes cumulative time in hours.

Baseline runs are drawn as lines. Non-baseline sweep variants are drawn as point clouds and labeled by the parameter change relative to the baseline.

## Audit the log before plotting

Treat plotting input as valid only if:

- `training.log` exists
- it contains at least one valid `Epoch ...` row
- parsed metric arrays are finite
- the referenced `Training_Directory` actually exists

If the plotting script fails, inspect the log directly:

```bash
sed -n '1,40p' <training_dir>/training.log
tail -n 25 <training_dir>/training.log
```

## When to use which plot path

- Use `--plot` on `scripts/training_script.py` when the user wants the repo's standard per-run `training.png`.
- Use `plotting/training_performance.py` when the user wants a cross-run PDF for the built-in sweep groups.
- If the user wants custom comparisons outside the built-in groups, start from the parsing logic in `plotting/training_performance.py` and the metric schema in `training.log`.
