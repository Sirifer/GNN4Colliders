#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-merged-shuffle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/scratchfs/cms/huangsuyun/GNN4Colliders/logs/hzg-prep-shuffle-%A_%a.out

set -euo pipefail

DATASET_NAME=$1
CONFIG_PATH=${2:-configs/hzg/ttH_vs_other_small.yaml}

source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate /publicfs/cms/user/huangsuyun/conda/envs/pytorch

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /cms/user/huangsuyun/GNN4Colliders/root_gnn_dgl

srun python scripts/prep_data.py \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET_NAME}" \
  --shuffle_mode \
  --chunk "${SLURM_ARRAY_TASK_ID}"
