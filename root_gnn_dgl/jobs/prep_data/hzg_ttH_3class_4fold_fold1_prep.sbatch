#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-prep-3class-fold1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_3class_4fold_fold1-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
mkdir -p ${DGL_CONF_PATH}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=configs/hzg/ttH_3class_4fold_fold1_baseline.yaml

cd ${REPO}

echo "=== Build + shuffle graphs for ttH (4 chunks) ==="
NCHUNKS_TTH=4
for chunk in $(seq 0 $((NCHUNKS_TTH - 1))); do
    echo "--- ttH raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --chunk ${chunk}
done
for chunk in $(seq 0 $((NCHUNKS_TTH - 1))); do
    echo "--- ttH shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Build + shuffle graphs for VH (4 chunks) ==="
NCHUNKS_VH=4
for chunk in $(seq 0 $((NCHUNKS_VH - 1))); do
    echo "--- VH raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset VH \
        --chunk ${chunk}
done
for chunk in $(seq 0 $((NCHUNKS_VH - 1))); do
    echo "--- VH shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset VH \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Build + shuffle graphs for ggF_VBF (8 chunks) ==="
NCHUNKS_GGF_VBF=8
for chunk in $(seq 0 $((NCHUNKS_GGF_VBF - 1))); do
    echo "--- ggF_VBF raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ggF_VBF \
        --chunk ${chunk}
done
for chunk in $(seq 0 $((NCHUNKS_GGF_VBF - 1))); do
    echo "--- ggF_VBF shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ggF_VBF \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Build + shuffle graphs for bkg (32 chunks) ==="
NCHUNKS_BKG=32
for chunk in $(seq 0 $((NCHUNKS_BKG - 1))); do
    echo "--- bkg raw chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset bkg \
        --chunk ${chunk}
done
for chunk in $(seq 0 $((NCHUNKS_BKG - 1))); do
    echo "--- bkg shuffle chunk ${chunk} ---"
    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset bkg \
        --shuffle_mode \
        --chunk ${chunk}
done

echo "=== Prep done for 3-class fold1 (baseline / shuffle / dropout share this cache) ==="
