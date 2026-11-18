#!/usr/bin/env python3
"""Generate the configs and sbatch jobs that score the v2 samples with the
v3-trained 4-class GNN, one model per v2 fold.

What is produced
----------------
configs/hzg/ttH_4class_on_v2_4fold_fold{0..3}.yaml
    `ttH_pt_only_4fold_fold{k}.yaml` (the v2 4-fold config: `raw_dir`
    `root_p2root_v2_pt_only`, `folding: n_folds 4, test [k]`) with

        Model.args.out_size        1 -> 4
        Training_Directory         trainings/hzg/ttH_4class_4fold_fold{k}
        Loss                       CrossEntropyLoss + Softmax(dim 1)

    The datasets, file lists, branches and the `folding` block are untouched,
    so the inputs are exactly the v2 events of `ttH_pt_only_4fold_fold{k}.yaml`
    and only the checkpoint comes from v3.

jobs/inference/hzg_ttH_4class_on_v2_4fold_fold{k}_inference.sbatch
    MC: root_p2root_v2_pt_only
     -> root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_fold{k}

jobs/inference/hzg_ttH_4class_on_v2_4fold_data_fold{k}_inference.sbatch
    Data: root_p2root_v2_pt_only_data
     -> root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_data_fold{k}/Data

jobs/inference/hzg_ttH_4class_on_v2_4fold_data_fix_fold{k}_inference.sbatch
    Data (fold-fixed): root_p2root_v2_pt_only_data_foldfix
     -> root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_data_fix_fold{k}/Data

    The shipped v2 Data files carry a broken `photon_eta_fold` (only 0 and 1,
    matching int32(photon_eta*10000) in 0 of 224 and 0 of 4346 events), so only
    two of the four folds could be selected from them.
    `tools/fix_v2_data_fold.py` rewrites the branch from `photon_eta`; these
    jobs score that corrected copy. Same events, same sideband total
    (9209), but all four parities exist. `tools/significance_4class_v2_bkg_4cat.py`
    reads the corrected directory with `--data-dir`.

    Same shape as `hzg_ttH_4class_4fold_fold0_inference.sbatch` and
    `hzg_ttH_4class_4fold_inference_data_fold0.sbatch`, with the 36 unrolled
    per-file blocks replaced by a loop over the same samples. The per-file
    command is identical: `--chunks 1 --chunkno 0 --write --tree
    merged_SR_Sideband --branch_name ttH_gnn_score`, same skip-if-exists
    logic. All four folds give the same 36 MC files (the `folding` block only
    changes which of them are training/test, not which are scored).

Fold handling - read this before using the output
-------------------------------------------------
`scripts/inference.py` does **not** apply `Datasets[...].folding`: it builds
the dataset from `Datasets[...]['args']` only, and `folding` is in the
dataset's `ignored_keys` (`root_gnn_base/dataset.py`). So, exactly like the
existing `root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold{k}`
directories, every output file contains **all** events of its input, all
scored by the fold-k model. Verified: `Signal/ttH/2022preEE.root` has 9390
events both before and after v3 fold-0 scoring.

The out-of-fold selection therefore happens downstream, with
`photon_eta_fold % 4 == k` — which is what `tools/eval_4class_on_v2.py` and
`tools/significance_4class_v2.py` already do. Do not assume that the fold-k
directory holds only fold-k events.

Eras: the 6 Run3 eras that the v2 MC samples and the existing v2 Data scoring
use (2022preEE ... 2025). `root_p2root_v2_pt_only_data` also contains
2026.root, but it is not in the v2 MC file lists and was not scored for the
binary model, so it is left out here to keep the two models on identical
events.

Usage (run from the repo root; writes nothing until you drop --check):
    python tools/make_v2_4class_on_v2_jobs.py --check
    python tools/make_v2_4class_on_v2_jobs.py
"""

import argparse
from pathlib import Path

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
DGL = REPO / "root_gnn_dgl"
CONFIG_DIR = DGL / "configs" / "hzg"
JOB_DIR = DGL / "jobs" / "inference"

NFOLD = 4
BASE_CONFIG = "ttH_pt_only_4fold_fold{fold}.yaml"
NEW_CONFIG = "ttH_4class_on_v2_4fold_fold{fold}.yaml"
# Data needs its own config: the v2 data files have `weight_lumi` and
# `weight_lumi_abs` but NOT `weight_lumi_avg`, and the fold configs carry
# `weight_var: weight_lumi_avg` (that is what killed jobs 632700/2/4/6 with
# uproot KeyInFileError). `ttH_pt_only_4fold_data.yaml` is the config the
# binary v2 data inference used: no `weight_var`, `tracking_info: []`.
BASE_DATA_CONFIG = "ttH_pt_only_4fold_data.yaml"
NEW_DATA_CONFIG = "ttH_4class_on_v2_4fold_data_fold{fold}.yaml"
TRAIN_DIR = str(REPO / "trainings" / "hzg" / "ttH_4class_4fold_fold{fold}")

LOSS_BLOCK = """Loss:
  module: torch.nn
  class: CrossEntropyLoss
  args: {}
  finish:
    module: torch.nn
    class: Softmax
    args: {dim: 1}
"""

PROCS = ["Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF",
         "Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100"]
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
        "2024", "2025"]

MC_IN = REPO / "root_p2root_v2_pt_only"
MC_OUT = REPO / "root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_fold{fold}"
DATA_IN = REPO / "root_p2root_v2_pt_only_data"
DATA_OUT = REPO / ("root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold"
                   "_data_fold{fold}")
# Fold-fixed v2 Data: `photon_eta_fold` recomputed as int32(photon_eta*10000)
# by tools/fix_v2_data_fold.py, because the shipped v2 Data branch only holds
# 0/1 (two parities instead of four). Same events, same sideband total.
DATA_FIX_IN = REPO / "root_p2root_v2_pt_only_data_foldfix"
DATA_FIX_OUT = REPO / ("root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold"
                       "_data_fix_fold{fold}")


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------
def make_config(fold, strict=True):
    """v2 4-fold config + 4-class model pointing at the v3 checkpoint."""
    src = CONFIG_DIR / BASE_CONFIG.format(fold=fold)
    text = src.read_text(encoding="utf-8")
    old_train = f"Training_Directory: {REPO}/trainings/hzg/" \
                f"ttH_pt_only_4fold_fold{fold}"

    edits = [
        (f"Training_Name: hzg_ttH_pt_only_4fold_fold{fold}",
         f"Training_Name: hzg_ttH_4class_on_v2_4fold_fold{fold}"),
        (old_train, f"Training_Directory: {TRAIN_DIR.format(fold=fold)}"),
        ("    out_size: 1", "    out_size: 4"),
        ("\nTraining:\n", "\n" + LOSS_BLOCK + "Training:\n"),
    ]
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            msg = f"{src}: expected 1 hit for {old!r}, found {n}"
            if strict:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        text = text.replace(old, new, 1)
    return CONFIG_DIR / NEW_CONFIG.format(fold=fold), text


def make_data_config(fold, strict=True):
    """Same, from the data config (no weight_var, no tracking_info)."""
    src = CONFIG_DIR / BASE_DATA_CONFIG
    text = src.read_text(encoding="utf-8")
    old_train = f"Training_Directory: {REPO}/trainings/hzg/" \
                f"ttH_pt_only_4fold_fold0"

    edits = [
        ("Training_Name: hzg_ttH_pt_only_4fold_data",
         f"Training_Name: hzg_ttH_4class_on_v2_4fold_data_fold{fold}"),
        (old_train, f"Training_Directory: {TRAIN_DIR.format(fold=fold)}"),
        ("    out_size: 1", "    out_size: 4"),
        ("\nTraining:\n", "\n" + LOSS_BLOCK + "Training:\n"),
    ]
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            msg = f"{src}: expected 1 hit for {old!r}, found {n}"
            if strict:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        text = text.replace(old, new, 1)
    if "weight_var" in text:
        raise RuntimeError(f"{src}: data config must not set weight_var")
    return CONFIG_DIR / NEW_DATA_CONFIG.format(fold=fold), text


# --------------------------------------------------------------------------
# sbatch
# --------------------------------------------------------------------------
SBATCH_HEAD = """#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-{jobname}-fold{fold}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/{logname}-%j.out

set -euo pipefail

mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs

export PYTHONUNBUFFERED=1
export DGLBACKEND=pytorch
export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config
export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
mkdir -p ${{DGL_CONF_PATH}} ${{MPLCONFIGDIR}}

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
CONFIG=configs/hzg/{config}

INPUT_DIR={input_dir}
OUTPUT_DIR={output_dir}

cd ${{REPO}}

echo "=== Inference: 4-class GNN (v3 fold{fold} model) on v2 {what} ==="
echo "Config: ${{CONFIG}}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

# scripts/inference.py ignores Datasets[...].folding, so every output file
# contains ALL events of its input, scored by the fold{fold} model. The
# out-of-fold parity is selected downstream with photon_eta_fold % 4 == {fold}.
echo "NOTE: full event list in the output; fold {fold} = photon_eta_fold % 4 == {fold}"
{extra_note}echo

{procs_line}ERAS="{eras}"
total={total}
count=0
"""

SBATCH_SCORE = """
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
    ${PYTHON} scripts/inference.py \\
        --config ${CONFIG} \\
        --target "${INPUT}" \\
        --destination "${OUTPUT}" \\
        --branch_name ttH_gnn_score \\
        --chunks 1 \\
        --chunkno 0 \\
        --write \\
        --tree merged_SR_Sideband
"""

SBATCH_TAIL = """
echo "=== done: ${count} new file(s) of ${total} expected ==="
"""


def render_sbatch(fold, what):
    """what is 'MC', 'Data' or 'DataFix' (fold-fixed v2 Data)."""
    is_mc = what == "MC"
    if is_mc:
        name = f"hzg_ttH_4class_on_v2_4fold_fold{fold}_inference.sbatch"
        jobname, logname = "infer-4class-v2", f"infer_4class_v2_fold{fold}"
        config = NEW_CONFIG.format(fold=fold)
        input_dir = str(MC_IN)
        output_dir = str(MC_OUT).format(fold=fold)
        total = len(PROCS) * len(ERAS)
        procs_line = f'PROCS="{" ".join(PROCS)}"\n'
        loop_head = 'for proc in ${PROCS}; do\n  for era in ${ERAS}; do\n' \
                    '    IN_REL="${proc}/${era}.root"\n' \
                    '    OUT_REL="${IN_REL}"\n'
        loop_close = '  done\ndone\n'
    else:
        suffix = "" if what == "Data" else "_fix"
        name = (f"hzg_ttH_4class_on_v2_4fold_data{suffix}_fold{fold}"
                "_inference.sbatch")
        jobname = "infer-4class-v2-data" if what == "Data" \
            else "infer-4class-v2-datafix"
        logname = f"infer_4class_v2_data{suffix}_fold{fold}"
        config = NEW_DATA_CONFIG.format(fold=fold)
        input_dir = str(DATA_IN if what == "Data" else DATA_FIX_IN)
        out_tmpl = DATA_OUT if what == "Data" else DATA_FIX_OUT
        output_dir = str(out_tmpl).format(fold=fold)   # Data/<era>.root
        total = len(ERAS)
        procs_line = ""
        loop_head = 'for era in ${ERAS}; do\n' \
                    '    IN_REL="${era}.root"\n' \
                    '    OUT_REL="Data/${era}.root"\n'
        loop_close = 'done\n'

    extra_note = "" if what != "DataFix" else (
        'echo "NOTE: input is root_p2root_v2_pt_only_data_foldfix — the v2 '
        'Data with photon_eta_fold rewritten as int32(photon_eta*10000)."\n'
        'echo "      The shipped v2 Data branch held only 0/1, so only two '
        'folds were selectable; the events and the sideband total are '
        'unchanged."\n'
    )
    head = SBATCH_HEAD.format(
        jobname=jobname, fold=fold, logname=logname,
        config=config, input_dir=input_dir,
        output_dir=output_dir,
        what={"MC": "MC", "Data": "Data",
              "DataFix": "Data (photon_eta_fold fixed)"}[what],
        total=total, procs_line=procs_line, eras=" ".join(ERAS),
        extra_note=extra_note,
    )
    text = head + loop_head + SBATCH_SCORE + loop_close + SBATCH_TAIL
    return JOB_DIR / name, text


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Write the v2-scoring configs and sbatch jobs for the "
                    "v3 4-class models (writes files, submits nothing).")
    parser.add_argument("--check", action="store_true",
                        help="print what would be written, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing outputs")
    args = parser.parse_args()

    outputs = []
    for fold in range(NFOLD):
        outputs.append(make_config(fold))
        outputs.append(make_data_config(fold))
        for what in ("MC", "Data", "DataFix"):
            outputs.append(render_sbatch(fold, what))

    for path, text in outputs:
        exists = path.exists()
        if args.check:
            print(f"{'would overwrite' if exists else 'would write'} "
                  f"{path.relative_to(REPO)}  ({len(text)} B)")
            continue
        if exists and not args.force:
            print(f"[skip] exists: {path.relative_to(REPO)} (use --force)")
            continue
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")

    if args.check:
        print(f"\n{len(outputs)} files, nothing written "
              f"(drop --check to write them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
