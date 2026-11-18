#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-prep-run3-v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_run3_relative_v2-%j.out

set -euo pipefail

# ============================================================
# Environment
# ============================================================

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config

mkdir -p ${DGL_CONF_PATH}


# ============================================================
# Paths
# ============================================================

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl

CONFIG=configs/hzg//ttH_relative_v2_fold0.yaml

# Must be consistent with chunks in YAML:
#   ttH   : chunks: 4
#   other : chunks: 32

TTH_CHUNKS=4
OTHER_CHUNKS=32


cd ${REPO}

echo "============================================================"
echo "HZgamma ttH vs other Run3 data preparation"
echo "Config: ${CONFIG}"
echo "ttH chunks   : ${TTH_CHUNKS}"
echo "other chunks : ${OTHER_CHUNKS}"
echo "============================================================"


# ============================================================
# Step 1: Build raw graphs for ttH
# ============================================================

echo
echo "=== Step 1: Build raw graphs for ttH ==="

for chunk in $(seq 0 $((TTH_CHUNKS - 1))); do

    echo
    echo "--- ttH raw chunk ${chunk}/${TTH_CHUNKS} ---"

    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --chunk ${chunk}

done


# ============================================================
# Step 2: Build raw graphs for other
# ============================================================

echo
echo "=== Step 2: Build raw graphs for other ==="

for chunk in $(seq 0 $((OTHER_CHUNKS - 1))); do

    echo
    echo "--- other raw chunk ${chunk}/${OTHER_CHUNKS} ---"

    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset other \
        --chunk ${chunk}

done


# ============================================================
# Step 3: Shuffle + prebatch ttH
# ============================================================

echo
echo "=== Step 3: Shuffle + prebatch ttH ==="

for chunk in $(seq 0 $((TTH_CHUNKS - 1))); do

    echo
    echo "--- ttH shuffle chunk ${chunk}/${TTH_CHUNKS} ---"

    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset ttH \
        --shuffle_mode \
        --chunk ${chunk}

done


# ============================================================
# Step 4: Shuffle + prebatch other
# ============================================================

echo
echo "=== Step 4: Shuffle + prebatch other ==="

for chunk in $(seq 0 $((OTHER_CHUNKS - 1))); do

    echo
    echo "--- other shuffle chunk ${chunk}/${OTHER_CHUNKS} ---"

    ${PYTHON} scripts/prep_data.py \
        --config ${CONFIG} \
        --dataset other \
        --shuffle_mode \
        --chunk ${chunk}

done


echo
echo "============================================================"
echo "All preprocessing finished successfully."
echo "============================================================"