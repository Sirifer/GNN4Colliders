#!/bin/bash
#SBATCH -N 1
#SBATCH -C "gpu&hbm80g"
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -o jobs/slurm/%j.out # STDOUT

ARGUMENTS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
mkdir -p jobs/slurm

echo "Arguments: ${ARGUMENTS[*]}"

source "$REPO_ROOT/setup/launch_image.sh" "$REPO_ROOT/jobs/training/podman/run_job_image.sh" "${ARGUMENTS[@]}"
