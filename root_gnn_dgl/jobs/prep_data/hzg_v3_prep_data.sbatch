#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-v3-prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/v3_prep-%j.out
# Time estimate: merge ~2.25h + pt_only ~2-2.5h for 100 files (Stitch is large).
#SBATCH --time=08:00:00

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
CODE=/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/code
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders

MERGED=${REPO}/root_p2root_v3_merged
PT_ONLY=${REPO}/root_p2root_v3_pt_only

echo "=== v3 4-class data preparation ==="
echo "Python:  ${PYTHON}"
echo "Merged output:  ${MERGED}"
echo "pt_only output: ${PT_ONLY}"
echo "Start: $(date)"
echo

# ----------------------------------------------------------------------------
# Step 1: merge v3 original samples -> root_p2root_v3_merged
#   Merges 5 categories' SR+Sideband per file, adds source_tree_tag,
#   weight_lumi_abs, photon_eta_fold, class_tag. Idempotent: existing output
#   files are skipped unless --force.
# ----------------------------------------------------------------------------
echo "=== Step 1: merge_trees_v3.py ==="
cd ${CODE}
${PYTHON} merge_trees_v3.py \
    --manifest merge_trees_v3_status.txt
echo "Step 1 done: $(date)"
echo

# Sanity check that merge actually produced files.
if [ ! -d "${MERGED}" ] || [ -z "$(find ${MERGED} -name '*.root' | head -1)" ]; then
    echo "ERROR: no merged .root files under ${MERGED}; aborting before pt_only."
    exit 1
fi
N_MERGED=$(find ${MERGED} -name '*.root' | wc -l)
echo "Merged files present: ${N_MERGED}"
echo

# ----------------------------------------------------------------------------
# Step 2: prepare pt_only inputs -> root_p2root_v3_pt_only
#   Adds _div_mllg derived branches + weight_lumi_avg (per-process mean weight
#   across all eras). Passes through all other branches incl. class_tag.
#   Refuses to overwrite an existing output dir (safety); remove it first to
#   rerun, or set FORCE_PT_ONLY=1 to let the job wipe and redo it.
# ----------------------------------------------------------------------------
if [ -d "${PT_ONLY}" ]; then
    if [ "${FORCE_PT_ONLY:-0}" = "1" ]; then
        echo "=== Step 2: FORCE_PT_ONLY=1 -> removing existing ${PT_ONLY} ==="
        rm -rf "${PT_ONLY}"
    else
        echo "=== Step 2: SKIP — ${PT_ONLY} already exists ==="
        echo "To rerun pt_only, submit with: sbatch --export=FORCE_PT_ONLY=1 ..."
        echo "Done (merge only): $(date)"
        exit 0
    fi
fi

echo "=== Step 2: prepare_pt_only_inputs_v3.py ==="
${PYTHON} prepare_pt_only_inputs_v3.py \
    --input-dir  ${MERGED} \
    --output-dir ${PT_ONLY}
echo "Step 2 done: $(date)"
echo

N_PT=$(find ${PT_ONLY} -name '*.root' | wc -l)
echo "=========================================================="
echo "v3 prep complete."
echo "  merged  : ${N_MERGED} files -> ${MERGED}"
echo "  pt_only : ${N_PT} files -> ${PT_ONLY}"
echo "  manifest: ${CODE}/merge_trees_v3_status.txt"
echo "  manifest: ${PT_ONLY}/manifest.json"
echo "Finish: $(date)"
echo "=========================================================="
