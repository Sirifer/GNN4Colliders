#!/bin/bash
# Three-step pipeline for CR_lowIDMVA Data (method-2 background estimate),
# 5 analysis categories only (ttHhad/ttHlep/untag/VH3l/VHpTmiss).
#
# Step 1: merge CR_lowIDMVA SR+SB of 5cat  -> root_p2root_v3_cr_merged/
# Step 2: add *_pt_div_mllg + weight_lumi_avg -> root_p2root_v3_cr_pt_only/
# Step 3: GNN inference (4 folds, same trained models) -> root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold{0-3}/
#
# Method-2 rationale: method 2 is fully data-driven. It takes the CR_lowIDMVA
# mass shape (CR SR region, source_tree_tag 0-4), scales CR data onto the
# nominal sideband data via the CR sideband integral (source_tree_tag 5-9),
# and reads the 120-130 yield of the scaled CR data as the background estimate.
# GNN scores are used downstream to bin events into signal-enriched regions
# for the final Z (per-category), but the core scale factor / transfer factor
# is GNN-independent.
#
# This pipeline does NOT touch the already-complete nominal MC and nominal
# Data pipelines. No retraining, no MC reprocessing.

set -eo pipefail

# conda's activate scripts (activate-binutils_linux-64.sh etc.) reference
# unbound vars like ADDR2LINE, which trips `set -u`. Activate with -u off,
# then re-enable it for the rest of the script.
source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
set -u

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders
REPO_GNN=${REPO}/root_gnn_dgl
TOOLS=${REPO}/tools/code

V3_SRC=/publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v3_Mll80to100

MERGED_DIR=${REPO}/root_p2root_v3_cr_merged
PREP_DIR=${REPO}/root_p2root_v3_cr_pt_only
SCORED_BASE=${REPO}/root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold

MERGE_SCRIPT=${TOOLS}/merge_cr_lowIDMVA_v3.py
PREP_SCRIPT=${TOOLS}/prepare_pt_only_inputs_v3.py

CR_TREE=merged_CR_lowIDMVA

# ---------------------------------------------------------------------------
# Step 1: merge CR_lowIDMVA SR+SB subtrees (5cat)
# ---------------------------------------------------------------------------
echo "=== Step 1: merge CR_lowIDMVA SR+SB (5cat, 11 eras) ==="

python -I "${MERGE_SCRIPT}" \
    --input-dir  "${V3_SRC}" \
    --output-dir "${MERGED_DIR}" \
    --manifest   "${REPO}/merge_cr_lowIDMVA_status.txt"

echo "[DONE] merge step -> ${MERGED_DIR}"

# ---------------------------------------------------------------------------
# Step 2: prepare_pt_only (add *_pt_div_mllg + weight_lumi_avg)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: prepare_pt_only_inputs_v3 (CR tree) ==="

if [ -d "${PREP_DIR}" ]; then
    echo "[INFO] ${PREP_DIR} already exists. Remove it to re-run this step."
else
    python -I "${PREP_SCRIPT}" \
        --input-dir  "${MERGED_DIR}" \
        --output-dir "${PREP_DIR}" \
        --tree       "${CR_TREE}"
    echo "[DONE] prepare -> ${PREP_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 3: inference, all 4 folds (reuse trained fold models)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: inference (4 folds, CR tree, reuse trained models) ==="

mkdir -p "${REPO}/logs"

for fold in 0 1 2 3; do
    CONFIG=${REPO_GNN}/configs/hzg/cr_lowIDMVA_ttH_4class_4fold_fold${fold}.yaml
    INPUT_DIR=${PREP_DIR}/Data/Data
    OUTPUT_DIR=${SCORED_BASE}_fold${fold}/Data/Data

    if [ ! -f "${CONFIG}" ]; then
        echo "[ERROR] missing CR config: ${CONFIG}"
        echo "        Generate it first: python ${TOOLS}/gen_cr_lowIDMVA_configs_jobs.py"
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "--- fold${fold} ---"
    for era in 2016preVFP 2016postVFP 2017 2018 2022preEE 2022postEE 2023preBPix 2023postBPix 2024 2025 2026; do
        SRC="${INPUT_DIR}/${era}.root"
        DST="${OUTPUT_DIR}/${era}.root"

        if [ ! -f "${SRC}" ]; then
            echo "[SKIP] missing prepared CR file: ${SRC}"
            continue
        fi
        if [ -f "${DST}" ]; then
            echo "[SKIP] already scored: ${era} (fold${fold})"
            continue
        fi

        echo "[RUN] fold${fold} ${era}"
        (
            cd "${REPO_GNN}"
            export PYTHONUNBUFFERED=1
            export DGLBACKEND=pytorch
            export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
            export MPLCONFIGDIR=${REPO}/cache/matplotlib
            python3 scripts/inference.py \
                --config      "${CONFIG}" \
                --target      "${SRC}" \
                --destination "${DST}" \
                --branch_name ttH_gnn_score \
                --chunks 1 \
                --chunkno 0 \
                --write \
                --clobber
        )
        echo "[DONE] fold${fold} ${era}"
    done
done

echo ""
echo "All done."
echo "  merged : ${MERGED_DIR}"
echo "  pt_only: ${PREP_DIR}"
echo "  scored : ${SCORED_BASE}_fold{0,1,2,3}"
