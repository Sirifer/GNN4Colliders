#!/bin/bash
# Three-step pipeline for all CR types (lowIDMVA/lowPTOverMllg/FSRgamma),
# 5 analysis categories only (ttHhad/ttHlep/untag/VH3l/VHpTmiss).
#
# Step 1: merge all 3 CR types, 5cat  → root_p2root_v2_cr_merged/
# Step 2: add *_pt_div_mllg           → root_p2root_v2_pt_only_cr_data/
# Step 3: GNN inference (fold0)       → root_p2root_v2_pt_only_cr_scored_fold0/

set -euo pipefail

source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders
REPO_GNN=${REPO}/root_gnn_dgl

DATA_DIR=${REPO}/root_p2root_v2/Data
MERGED_DIR=${REPO}/root_p2root_v2_cr_merged
PREP_DIR=${REPO}/root_p2root_v2_pt_only_cr_data
SCORED_DIR=${REPO}/root_p2root_v2_pt_only_cr_scored_fold0

CONFIG=${REPO_GNN}/configs/hzg/ttH_pt_only_data_fold0.yaml

ERAS=(2022preEE 2022postEE 2023preBPix 2023postBPix 2024 2025)

# ---------------------------------------------------------------------------
# Step 1: merge all 3 CR types → merged_CR
# ---------------------------------------------------------------------------
echo "=== Step 1: merge CR subtrees ==="

python -I "${REPO}/tools/code/merge_cr_trees.py" \
    --input-dir  "${DATA_DIR}" \
    --output-dir "${MERGED_DIR}"

echo "[DONE] merge step"

# ---------------------------------------------------------------------------
# Step 2: prepare (add *_pt_div_mllg)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: prepare_pt_only_inputs ==="

if [ -d "${PREP_DIR}" ]; then
    echo "[INFO] ${PREP_DIR} already exists, skipping prepare step."
else
    python -I "${REPO}/tools/code/prepare_pt_only_inputs.py" \
        --input-dir  "${MERGED_DIR}" \
        --output-dir "${PREP_DIR}" \
        --tree       merged_CR
    echo "[DONE] prepare"
fi

# ---------------------------------------------------------------------------
# Step 3: inference
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: inference (fold0) ==="

mkdir -p "${SCORED_DIR}"

cd "${REPO_GNN}"

for era in "${ERAS[@]}"; do
    SRC="${PREP_DIR}/${era}.root"
    DST="${SCORED_DIR}/${era}.root"

    if [ ! -f "${SRC}" ]; then
        echo "[SKIP] missing prepared file: ${SRC}"
        continue
    fi

    if [ -f "${DST}" ]; then
        echo "[SKIP] already scored: ${DST}"
        continue
    fi

    echo "[RUN] ${era}"
    python3 scripts/inference.py \
        --config      "${CONFIG}" \
        --target      "${SRC}" \
        --destination "${DST}" \
        --tree        merged_CR \
        --branch_name ttH_gnn_score \
        --write \
        --clobber

    echo "[DONE] ${era}"
done

echo ""
echo "All done. Scored files in: ${SCORED_DIR}"
