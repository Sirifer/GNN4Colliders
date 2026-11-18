#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-infer-4class-v2-datafix-fold0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_4class_v2_data_fix_fold0-%j.out

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
CONFIG=configs/hzg/ttH_4class_on_v2_4fold_data_fold0.yaml

INPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_data_foldfix
OUTPUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_data_fix_fold0

cd ${REPO}

echo "=== Inference: 4-class GNN (v3 fold0 model) on v2 Data (photon_eta_fold fixed) ==="
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

# scripts/inference.py ignores Datasets[...].folding, so every output file
# contains ALL events of its input, scored by the fold0 model. The
# out-of-fold parity is selected downstream with photon_eta_fold % 4 == 0.
echo "NOTE: full event list in the output; fold 0 = photon_eta_fold % 4 == 0"
echo "NOTE: input is root_p2root_v2_pt_only_data_foldfix — the v2 Data with photon_eta_fold rewritten as int32(photon_eta*10000)."
echo "      The shipped v2 Data branch held only 0/1, so only two folds were selectable; the events and the sideband total are unchanged."
echo

ERAS="2022preEE 2022postEE 2023preBPix 2023postBPix 2024 2025"
total=6
count=0
for era in ${ERAS}; do
    IN_REL="${era}.root"
    OUT_REL="Data/${era}.root"

    INPUT="${INPUT_DIR}/${IN_REL}"
    OUTPUT="${OUTPUT_DIR}/${OUT_REL}"
    if [ ! -f "${INPUT}" ]; then
        echo "[skip] missing: ${IN_REL}"
        continue
    fi
    if [ -f "${OUTPUT}" ]; then
        echo "[skip] exists: ${OUT_REL}"
        continue
    fi
    mkdir -p "$(dirname "${OUTPUT}")"
    count=$((count + 1))
    echo "[${count}/${total}] scoring ${IN_REL}"
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

echo "=== done: ${count} new file(s) of ${total} expected ==="
