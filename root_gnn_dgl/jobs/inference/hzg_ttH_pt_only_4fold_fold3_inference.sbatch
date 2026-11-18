#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-pt-only-4fold-fold3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_pt_only_4fold_fold3-%j.out

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

CONFIG=configs/hzg/ttH_pt_only_4fold_fold3.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold3

cd ${REPO}

echo "=== Inference: ttH pt_only 4fold fold3 ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

SAMPLES=(
    "Signal/ttH"
    "Signal/VH"
    "Signal/ggF"
    "Signal/VBF"
    "Bkg_MC/TT"
    "Bkg_MC/DYGto2LG_10to100"
)

YEARS=(
    "2022preEE"
    "2022postEE"
    "2023preBPix"
    "2023postBPix"
    "2024"
    "2025"
)

total=$((${#SAMPLES[@]} * ${#YEARS[@]}))
count=0

for SAMPLE in "${SAMPLES[@]}"; do
    for YEAR in "${YEARS[@]}"; do

        count=$((count + 1))

        REL="${SAMPLE}/${YEAR}.root"
        INPUT="${INPUT_DIR}/${REL}"
        OUTPUT="${OUTPUT_DIR}/${REL}"

        if [ ! -f "${INPUT}" ]; then
            echo "[${count}/${total}] Missing: ${REL}"
            continue
        fi

        mkdir -p "$(dirname "${OUTPUT}")"

        if [ -f "${OUTPUT}" ]; then
            echo "[${count}/${total}] Skip (exists): ${REL}"
            continue
        fi

        echo "[${count}/${total}] Scoring: ${REL}"

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
done

echo "=== Done ==="
