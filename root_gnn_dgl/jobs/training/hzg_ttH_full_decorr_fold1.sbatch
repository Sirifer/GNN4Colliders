#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-train-full-decorr-fold1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/train_full_decorr_fold1-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs
mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/ttH_full_decorr_fold1

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p ${DGL_CONF_PATH}
mkdir -p ${MPLCONFIGDIR}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=configs/hzg/ttH_full_decorr_fold1.yaml

cd ${REPO}

echo "=== Training: ttH full decorr fold1 ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

srun ${PYTHON} scripts/training_script.py \
    --config ${CONFIG} \
    --preshuffle \
    --nocompile \
    --lazy \
    --restart

echo "=== Done ==="
