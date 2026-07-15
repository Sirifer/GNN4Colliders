#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 15:00:00
#SBATCH -o jobs/slurm/%j.out # STDOUT

CONFIG=$1
shift
ARGUMENTS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
mkdir -p jobs/slurm

eval "$(conda shell.bash hook)"
conda activate "${ROOT_GNN_CONDA_ENV:-dgl}"

COMMAND=(
    "$REPO_ROOT/scripts/training_script.py"
    "${ARGUMENTS[@]}"
    "--preshuffle"
    "--nocompile"
    "--lazy"
    "--config"
    "$CONFIG"
)

echo "Running my script now"
echo "Executing: python -u ${COMMAND[*]}"
python -u "${COMMAND[@]}"
echo "Done"
