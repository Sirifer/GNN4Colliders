#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-TL-Graph-FT-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_TL_Graph_FT_data-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p ${DGL_CONF_PATH} ${MPLCONFIGDIR}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl

CONFIG=configs/hzg/ttH_TL_Graph_FT_data_fold0.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_data
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_TL_Graph_FT_fold0/Data

cd ${REPO}

echo "=== Inference (Data): ttH TL_Graph_FT fold0 model ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

YEARS=(
    "2022preEE"
    "2022postEE"
    "2023preBPix"
    "2023postBPix"
    "2024"
    "2025"
)

total=${#YEARS[@]}
count=0

for YEAR in "${YEARS[@]}"; do

    count=$((count + 1))

    INPUT="${INPUT_DIR}/${YEAR}.root"
    OUTPUT="${OUTPUT_DIR}/${YEAR}.root"

    if [ ! -f "${INPUT}" ]; then
        echo "[${count}/${total}] Missing: ${YEAR}.root"
        continue
    fi

    mkdir -p "$(dirname "${OUTPUT}")"

    if [ -f "${OUTPUT}" ]; then
        echo "[${count}/${total}] Skip (exists): ${YEAR}.root"
        continue
    fi

    echo "[${count}/${total}] Scoring: ${YEAR}.root"

    ${PYTHON} scripts/inference.py \
        --config ${CONFIG} \
        --target "${INPUT}" \
        --destination "${OUTPUT}" \
        --branch_name ttH_gnn_score \
        --chunks 1 \
        --chunkno 0 \
        --write \
        --tree merged_SR_Sideband

done

echo "=== Done ==="
