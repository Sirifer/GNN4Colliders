#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-cr-fold1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_cr_lowIDMVA_4fold_fold1-%j.out

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
CONFIG=configs/hzg/cr_lowIDMVA_ttH_4class_4fold_fold1.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_cr_pt_only/Data/Data
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold1/Data/Data

cd ${REPO}

echo "=== Inference: CR_lowIDMVA ttH 4class 4fold fold1 ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

total=11
count=0

REL="2016preVFP.root"
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
    --tree merged_CR_lowIDMVA

REL="2016postVFP.root"
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
    --tree merged_CR_lowIDMVA

REL="2017.root"
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
    --tree merged_CR_lowIDMVA

REL="2018.root"
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
    --tree merged_CR_lowIDMVA

REL="2022preEE.root"
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
    --tree merged_CR_lowIDMVA

REL="2022postEE.root"
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
    --tree merged_CR_lowIDMVA

REL="2023preBPix.root"
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
    --tree merged_CR_lowIDMVA

REL="2023postBPix.root"
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
    --tree merged_CR_lowIDMVA

REL="2024.root"
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
    --tree merged_CR_lowIDMVA

REL="2025.root"
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
    --tree merged_CR_lowIDMVA

REL="2026.root"
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
    --tree merged_CR_lowIDMVA

echo "=== Done ==="
