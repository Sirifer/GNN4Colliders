#!/bin/bash
# Run GNN inference on CR_lowIDMVA data.
# Reads from:  /publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v2/Data/Data/
# Writes to:   /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_cr_lowIDMVA_scored_fold0/Data/
#
# Uses fold0 model only (fold_var=photon_eta, parity 0 events).
# Output branch name matches nominal scored files: ttH_gnn_score
# (sanitize_branch_name("hzg_ttH_vs_other_run3_fold0") -> "hzg_ttH_vs_other_run3_fold0_score",
#  but we override with --branch_name to keep it consistent with existing Data files.)

set -euo pipefail

source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=${REPO}/configs/hzg/ttH_vs_other_run3_v2_data.yaml
CR_DIR=/publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v2/Data/Data
OUT_DIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_cr_lowIDMVA_scored_fold0/Data

mkdir -p "${OUT_DIR}"

ERAS=(2022preEE 2022postEE 2023preBPix 2023postBPix 2024 2025)

cd "${REPO}"

for era in "${ERAS[@]}"; do
    SRC="${CR_DIR}/${era}.root"
    DST="${OUT_DIR}/${era}.root"

    if [ ! -f "${SRC}" ]; then
        echo "[SKIP] missing source: ${SRC}"
        continue
    fi

    if [ -f "${DST}" ]; then
        echo "[SKIP] already exists: ${DST}"
        continue
    fi

    echo "[RUN] ${era}"
    python3 scripts/inference.py \
        --config   "${CONFIG}" \
        --target   "${SRC}" \
        --destination "${DST}" \
        --tree     CR_lowIDMVA__inclusive \
        --branch_name ttH_gnn_score \
        --write \
        --clobber

    echo "[DONE] ${era}"
done

echo ""
echo "All done. Output in: ${OUT_DIR}"
