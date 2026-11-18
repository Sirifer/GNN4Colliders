#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-run3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_run3-%j.out

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
CONFIG=configs/hzg/higgs_vs_bkg_run2run3.yaml
INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_merged
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored_higgsvsbkg_run2run3

cd ${REPO}

echo "=== Inference: ttH_vs_other_run23_v2 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

total=$(find ${INPUT_DIR} -name "*.root" | wc -l)
count=0

while IFS= read -r INPUT; do
    count=$((count + 1))

    # 镜像目录结构
    REL=${INPUT#${INPUT_DIR}/}
    OUTPUT=${OUTPUT_DIR}/${REL}
    mkdir -p "$(dirname ${OUTPUT})"

    # 已存在则跳过
    if [ -f "${OUTPUT}" ]; then
        echo "[${count}/${total}] Skip (exists): ${REL}"
        continue
    fi

    echo "[${count}/${total}] Scoring: ${REL}"
    ${PYTHON} scripts/inference.py \
        --config ${CONFIG} \
        --target ${INPUT} \
        --destination ${OUTPUT} \
        --branch_name ttH_gnn_score \
        --chunks 1 \
        --chunkno 0 \
        --write \
        --tree merged_SR_Sideband

done < <(find ${INPUT_DIR} -name "*.root" | sort)

echo "=== Done: ${count} files processed ==="
