#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-4class-data-minloss-fold2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_4class_minloss_data_fold2-%j.out

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
CONFIG=configs/hzg/ttH_4class_4fold_fold2.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only
# Data IS produced by prepare_pt_only_inputs_v3.py (it has weight_lumi=1
# in v3, so it goes through the same pt_only pipeline as MC).
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_minloss_fold2

cd ${REPO}

echo "=== Inference: ttH 4class 4fold MINLOSS Data fold2 ==="

total=11
count=0

REL="Data/Data/2016preVFP.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2016postVFP.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2017.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2018.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2022preEE.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2022postEE.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2023preBPix.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2023postBPix.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2024.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2025.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

REL="Data/Data/2026.root"
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
    --var Test_Loss \
    --mode min \
    --chunks 1 \
    --chunkno 0 \
    --write \
    --tree merged_SR_Sideband

echo "=== Done ==="
