---
name: root-gnn-dgl-training
description: Use when the user asks to train, finetune, resume, submit, queue-check, log-check, or validate training runs in root_gnn_dgl, including scripts/training_script.py, sqs, jobs/slurm/<JOBID>.out, training.log, checkpoints, and scripts/generate_multiclass_finetuning_configs.py.
---

# root-gnn-dgl-training

Use this skill for any training stage in the repo, from launch through monitoring and artifact review.

## Run locally

Run from the repo root:

```bash
python scripts/training_script.py --config <config.yaml> --preshuffle --nocompile --lazy
```

This matches the README, `run_demo.sh`, and the podman job wrappers.

## Why these defaults

- `--preshuffle` uses the saved prebatched graph files created during data prep.
- `--nocompile` is recommended by the README because compiled mode requires padded graphs at prep time.
- `--lazy` matches the common dataset classes used by the shipped configs.

## Common runtime modes

- `--restart` starts from scratch instead of resuming from the last checkpoint.
- Without `--restart`, the script resumes from the last `model_epoch_<n>.pt` it finds in `Training_Directory`.
- `--evaluate <epoch>` skips training and evaluates a specific checkpoint.
- `--plot` regenerates `training.png` from `training.log`.
- `--directory <suffix>` appends a suffix to `Training_Directory`.
- `--cpu`, `--multigpu`, `--multinode`, `--statistics`, `--seed`, and `--abs` are available when needed.

## Run multiple local trainings

Use the single-node launcher when queueing many config files on the current node:

```bash
python jobs/training/run_parallel_trainings.py <config-or-directory> --split <K/N>
```

Useful examples:

```bash
python jobs/training/run_parallel_trainings.py configs/run_3_ttH --split 2/2
python jobs/training/run_parallel_trainings.py configs/run_3_ttH --split 2/2 --test
```

Launcher behavior:

- Discovers `.yaml` files from each target directory, or accepts explicit config files.
- Uses fixed local GPU slots `0,1,2,3` via `CUDA_VISIBLE_DEVICES`.
- Runs `scripts/training_script.py --config <config> --preshuffle --nocompile --lazy`.
- Forwards unknown args to `training_script.py`, such as `--restart` or `--seed 7`.
- Rejects configs with duplicate `Training_Directory` values.
- `--split K/N` selects configs by deterministic index modulo `N`; use `--split 1/2` and `--split 2/2` for two complementary halves.
- `--test` prints the launch plan without starting training.
- Logs go under `jobs/slurm/parallel_training_logs/<timestamp>/`.

## Submit on Perlmutter

Prefer the podman path:

```bash
sbatch jobs/training/podman/run_job.sh <config>
```

The job wrappers derive the repo root from their own location. `jobs/training/podman/submit.sh` and `jobs/training/conda/submit.sh` accept config paths as arguments; without arguments they run their built-in default sweeps. Set `SLURM_ACCOUNT` when the cluster requires an account:

```bash
SLURM_ACCOUNT=atlas sbatch jobs/training/podman/run_job.sh configs/run_3_ttH/scratch.yaml
bash jobs/training/podman/submit.sh configs/run_3_ttH/scratch.yaml configs/run_3_ttH/finetuning.yaml
```

For the conda wrapper, set `ROOT_GNN_CONDA_ENV` if the environment is not named `dgl`.

For distributed training, pass `--multinode` and launch under an environment that sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`.

## Preconditions

- If you use `--preshuffle`, run data preparation first and confirm the graph artifacts exist.
- For finetuning configs, verify that `Model.args.pretraining_path` points to an existing checkpoint before launching training.
- For multinode runs, pass `--multinode` and launch under the relevant distributed job environment.

## Monitor queue and logs

Check queue state:

```bash
sqs -u "$USER"
sqs -u "$USER" | rg "<pattern>"
```

Useful interpretations:

- `PD` means pending
- `R` means running
- `START_TIME N/A` with reason `Priority` means queued normally, not broken

Once a job has a `JOBID`, inspect:

```bash
sed -n '1,80p' jobs/slurm/<JOBID>.out
tail -n 80 jobs/slurm/<JOBID>.out
rg -n "Traceback|Error|Exception|Epoch|Epoch Done|Early Termination|Done" jobs/slurm/<JOBID>.out
```

Healthy training logs usually show:

- the `Executing: python -u ... scripts/training_script.py ...` line
- dataset cache loads
- repeated `Epoch ... | LR ... | Loss ... | Accuracy ... | Test_Loss ... | Test_AUC ... | Time ... s`
- `Epoch Done.`
- `Num batches trained = ...`
- valid completion via `Done`, sometimes after `Early Termination at Epoch ...`

Early stopping is a normal completion mode in this repo.

## Audit training artifacts

Training writes into `Training_Directory`:

- `config.yaml`
- `model_epoch_<n>.pt`
- `model_epoch_<n>.npz`
- `training.log`
- `training.png`

Primary checks:

```bash
sed -n '1,40p' <training_dir>/training.log
tail -n 25 <training_dir>/training.log
```

Treat the run as healthy only if:

- epoch numbers increase monotonically
- `Loss`, `Test_Loss`, and `Test_AUC` stay finite
- the latest logged epoch has a matching `model_epoch_<n>.pt`
- the run produced real epoch rows rather than stopping before training started

If `training.log` grows but checkpoints stop appearing, suspect a save-path or filesystem issue.

Use `python plotting/training_performance.py` or the `root-gnn-dgl-plotting` skill when you want consolidated sweep-level PDFs instead of a single-run `training.png`.

## Generate finetuning configs

Use this when you want to derive `configs/higgs_production/multiclass_finetuning/*.yaml` from completed multiclass pretraining runs:

```bash
python scripts/generate_multiclass_finetuning_configs.py
```

Before using the generated configs, verify that the chosen best-epoch checkpoint paths still exist.
