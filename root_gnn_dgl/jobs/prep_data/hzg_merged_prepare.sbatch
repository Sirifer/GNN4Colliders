#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-merged-base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/merge-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1

/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python \
    /scratchfs2/cms/huangsuyun/GNN4Colliders/tools/merge_trees_all.py

# cd /cms/user/huangsuyun/GNN4Colliders/root_gnn_dgl

# srun python scripts/prep_data.py \
#   --config "${CONFIG_PATH}" \
#   --dataset "${DATASET_NAME}" \
#   --chunk "${SLURM_ARRAY_TASK_ID}"
