#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-prep-4fold-fold3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_4fold_fold3-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
mkdir -p ${DGL_CONF_PATH}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=configs/hzg/ttH_pt_only_4fold_fold3.yaml

TTH_CHUNKS=4
OTHER_CHUNKS=32

cd ${REPO}

echo "=== Step 1: Build raw graphs for ttH ==="
for chunk in $(seq 0 $((TTH_CHUNKS - 1))); do
    echo "--- ttH raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --chunk ${chunk}
done

echo "=== Step 2: Build raw graphs for other ==="
for chunk in $(seq 0 $((OTHER_CHUNKS - 1))); do
    echo "--- other raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset other \
        --chunk ${chunk}
done

echo "=== Step 3: Shuffle + prebatch ttH ==="
for chunk in $(seq 0 $((TTH_CHUNKS - 1))); do
    echo "--- ttH shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Step 4: Shuffle + prebatch other ==="
for chunk in $(seq 0 $((OTHER_CHUNKS - 1))); do
    echo "--- other shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset other \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Prep done for fold3 ==="
