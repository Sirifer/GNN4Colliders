#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-prep-higgs-bkg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_higgs_vs_bkg_run3-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
mkdir -p ${DGL_CONF_PATH}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=configs/hzg/higgs_vs_bkg_run3_v2_fold0.yaml

cd ${REPO}

echo "=== Step 1: Build raw graphs for higgs (16 chunks) ==="
for chunk in $(seq 0 15); do
    echo "--- higgs chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset higgs \
        --chunk ${chunk}
done

echo "=== Step 2: Build raw graphs for bkg (8 chunks) ==="
for chunk in $(seq 0 7); do
    echo "--- bkg chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset bkg \
        --chunk ${chunk}
done

echo "=== Step 3: Shuffle+prebatch higgs ==="
for chunk in $(seq 0 15); do
    echo "--- higgs shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset higgs \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Step 4: Shuffle+prebatch bkg ==="
for chunk in $(seq 0 7); do
    echo "--- bkg shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset bkg \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== All done ==="