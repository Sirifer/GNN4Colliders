#!/bin/bash

date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_JOB="$REPO_ROOT/jobs/training/podman/run_job.sh"
SBATCH_ACCOUNT_ARGS=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then
  SBATCH_ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

default_configs=(
  # "configs/multiclass_pretraining/baseline.yaml"
  # "configs/multiclass_pretraining/pretraining_batch_size/multiclass_bs_4096.yaml"
  # "configs/multiclass_pretraining/pretraining_hid_size/multiclass_hid_256.yaml"
  # "configs/multiclass_pretraining/pretraining_lr/multiclass_lr_1e2.yaml"
  # "configs/multiclass_pretraining/pretraining_n_layers/multiclass_layers_6.yaml"

  # "configs/multiclass_pretraining/pretraining_batch_size/multiclass_bs_2048.yaml"
  # "configs/multiclass_pretraining/pretraining_hid_size/multiclass_hid_128.yaml"
  # "configs/multiclass_pretraining/pretraining_lr/multiclass_lr_1e3.yaml"
  # "configs/multiclass_pretraining/pretraining_n_layers/multiclass_layers_5.yaml"

  # "configs/higgs_production/baseline.yaml"
  # "configs/higgs_production/higgs_production_batch_size/higgs_production_bs_4096.yaml"
  # "configs/higgs_production/higgs_production_hid_size/higgs_production_hid_256.yaml"
  # "configs/higgs_production/higgs_production_lr/higgs_production_lr_1e2.yaml"
  # "configs/higgs_production/higgs_production_n_layers/higgs_production_layers_6.yaml"

  # "configs/higgs_production/higgs_production_batch_size/higgs_production_bs_2048.yaml"
  # "configs/higgs_production/higgs_production_hid_size/higgs_production_hid_128.yaml"
  # "configs/higgs_production/higgs_production_lr/higgs_production_lr_1e3.yaml"
  # "configs/higgs_production/higgs_production_n_layers/higgs_production_layers_5.yaml"
  # "configs/higgs_production/baseline2.yaml"
  # "configs/higgs_production/baseline3.yaml"
  # "configs/higgs_production/baseline4.yaml"
  # "configs/higgs_production/baseline5.yaml"
  "configs/higgs_production/multiclass_finetuning/baseline.yaml"
  "configs/higgs_production/multiclass_finetuning/baseline_lr_1e4.yaml"
  "configs/higgs_production/multiclass_finetuning/baseline_lr_1e6.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_128.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_128_lr_1e4.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_128_lr_1e6.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_256.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_256_lr_1e4.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_hid_256_lr_1e6.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_layers_6.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_layers_6_lr_1e4.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_layers_6_lr_1e6.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_lr_1e3.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_lr_1e3_lr_1e4.yaml"
  "configs/higgs_production/multiclass_finetuning/multiclass_lr_1e3_lr_1e6.yaml"


)

if [ "$#" -gt 0 ]; then
  configs=("$@")
else
  configs=("${default_configs[@]}")
fi

counter=0

hours=24
time="${hours}:00:00"

for job in "${configs[@]}"
do
  sbatch "${SBATCH_ACCOUNT_ARGS[@]}" --job-name="$job" --time="$time" "$RUN_JOB" "$job"
  ((counter++))
done

echo "Total jobs submitted: $counter"
