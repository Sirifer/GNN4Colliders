#!/bin/bash

date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_JOB="$REPO_ROOT/jobs/training/conda/run_job.sh"
SBATCH_ACCOUNT_ARGS=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then
  SBATCH_ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

default_configs=(
  "configs/stats_all/ttH_CP_even_vs_odd.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_2048.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_4096.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_8192.yaml"
)

if [ "$#" -gt 0 ]; then
  configs=("$@")
else
  configs=("${default_configs[@]}")
fi

counter=0

hours=12
time="${hours}:00:00"

for job in "${configs[@]}"
do
  sbatch "${SBATCH_ACCOUNT_ARGS[@]}" --job-name="$job" --time="$time" "$RUN_JOB" "$job"
  ((counter++))
done

echo "Total jobs submitted: $counter"
