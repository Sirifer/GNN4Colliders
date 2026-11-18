#!/bin/bash
# submit_merge.sh
#
# Generates a filelist of all ROOT files under root_p2root_v2/,
# then submits them as a SLURM array job via hzg_merge_trees.sbatch.
# Output goes to root_p2root_v2_merged/ with identical subdirectory structure.
# Each output file contains only the merged_SR_Sideband tree.
#
# Usage:
#   bash tools/submit_merge.sh            # submit
#   bash tools/submit_merge.sh --dry-run  # preview only

set -euo pipefail

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders
DATA_ROOT=${REPO}/root_p2root_v2
SBATCH=${REPO}/root_gnn_dgl/jobs/prep_data/hzg_merge_trees.sbatch
FILELIST=${REPO}/tools/merge_filelist.txt
LOGDIR=${REPO}/logs

mkdir -p "$LOGDIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# ------------------------------------------------------------------
# Build filelist: Signal + Bkg_MC (all eras)
# ------------------------------------------------------------------
echo "Scanning $DATA_ROOT ..."
find "$DATA_ROOT/Signal" "$DATA_ROOT/Bkg_MC" \
     -name "*.root" | sort > "$FILELIST"

N=$(wc -l < "$FILELIST")
echo "Found $N ROOT files -> $FILELIST"

if [[ $N -eq 0 ]]; then
    echo "ERROR: no ROOT files found under $DATA_ROOT"
    exit 1
fi

ARRAY_MAX=$((N - 1))

echo ""
echo "Input:   $DATA_ROOT"
echo "Output:  ${DATA_ROOT/root_p2root_v2/root_p2root_v2_merged}"
echo "Sbatch:  $SBATCH"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "[dry-run] Not submitting. First 10 files:"
    head -10 "$FILELIST"
    exit 0
fi

sbatch "$SBATCH" "$FILELIST"
echo "Submitted 1 job covering $N files (~8h estimated)."
