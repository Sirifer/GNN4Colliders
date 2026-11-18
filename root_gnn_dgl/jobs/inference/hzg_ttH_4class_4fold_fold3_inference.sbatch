#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-4class-fold3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_4class_4fold_fold3-%j.out

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
CONFIG=configs/hzg/ttH_4class_4fold_fold3.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold3

cd ${REPO}

echo "=== Inference: ttH 4class 4fold fold3 ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

total=89
count=0

REL="Signal/ttH/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ttH/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VH/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/ggF/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Signal/VBF/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/TT/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_10to100/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_100to200/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_100to200/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_100to200/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_100to200/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_200to400/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_200to400/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_200to400/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_200to400/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_400to600/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_400to600/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_400to600/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_400to600/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_600/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_600/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_600/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYGto2LG_600/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2016preVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2016postVFP.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2017.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2018.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2022preEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2022postEE.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2023preBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2023postBPix.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2024.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2025.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

REL="Bkg_MC/DYJetsToLL_Stitch/2026.root"
INPUT="${INPUT_DIR}/${REL}"
OUTPUT="${OUTPUT_DIR}/${REL}"
if [ ! -f "${INPUT}" ]; then
    echo "[skip] missing: ${REL}"
    continue
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [ -f "${OUTPUT}" ]; then
    echo "[skip] exists: ${REL}"
    continue
fi
count=$((count + 1))
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

echo "=== Done ==="
