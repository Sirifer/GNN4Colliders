#!/usr/bin/env python3
"""
gen_4class_4fold_configs_jobs.py

Generate the 4-class, 4-fold configs and SLURM job scripts for the v3 pt_only
setup. Produces, under root_gnn_dgl/:

  configs/hzg/ttH_4class_4fold_fold{0,1,2,3}.yaml
  jobs/prep_data/hzg_ttH_4class_4fold_fold{0,1,2,3}_prep.sbatch
  jobs/training/hzg_ttH_4class_4fold_fold{0,1,2,3}.sbatch
  jobs/inference/hzg_ttH_4class_4fold_fold{0,1,2,3}_inference.sbatch
  jobs/inference/hzg_ttH_4class_4fold_inference_data.sbatch

4-class definition (label values feed the multiclass CrossEntropyLoss path):
    ttH       label 1
    VH        label 2
    ggF+VBF   label 3   (ggF and VBF files merged into one dataset)
    bkg       label 0   (TT + 5 DYG mass points + DYJetsToLL_Stitch)

All eras present in v3 (Run2+Run3: 2016preVFP..2026) are used per sample;
files that do not exist for a sample are simply absent from that dataset's
file list (e.g. DYG high-mass points only have 4 Run3 eras).

Fold rotation (same as the old 4fold configs):
    fold0: test=[0], train=[1,2,3]
    fold1: test=[1], train=[0,2,3]
    fold2: test=[2], train=[0,1,3]
    fold3: test=[3], train=[0,1,2]

Input scheme: same as old pt_only 4fold -- lepton (0-5) and photon pT use the
_div_mllg derived branches; jet (0-5), bjet (0-1), MET use raw pT.

Usage:
    python gen_4class_4fold_configs_jobs.py            # write all files
    python gen_4class_4fold_configs_jobs.py --dry-run  # print, write nothing
"""

import argparse
import os
from pathlib import Path

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
DGL = REPO / "root_gnn_dgl"

V3_INPUT_DIR = REPO / "root_p2root_v3_pt_only"
GRAPH_DIR = REPO / "graphs" / "hzg" / "ttH_4class_4fold"
TRAIN_DIR_BASE = REPO / "trainings" / "hzg" / "ttH_4class_4fold"
SCORED_DIR_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold"

PYTHON_BIN = "/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python"

# v3 source sample root (used to enumerate era files per sample)
V3_SRC = Path("/publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v3_Mll80to100")

# eras present in v3 Data (Run2+Run3). We list each sample's actually-existing
# era files below by scanning V3_SRC, so missing eras per sample are handled.
ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

# 4-class dataset definitions: (dataset_name, label, [sample subdirs under V3_SRC])
DATASETS = [
    ("ttH",      1, ["Signal/ttH"]),
    ("VH",       2, ["Signal/VH"]),
    ("ggF_VBF",  3, ["Signal/ggF", "Signal/VBF"]),
    ("bkg",      0, [
        "Bkg_MC/TT",
        "Bkg_MC/DYGto2LG_10to100",
        "Bkg_MC/DYGto2LG_100to200",
        "Bkg_MC/DYGto2LG_200to400",
        "Bkg_MC/DYGto2LG_400to600",
        "Bkg_MC/DYGto2LG_600",
        "Bkg_MC/DYJetsToLL_Stitch",
    ]),
]

# 4-fold rotation: test fold -> train folds
FOLD_PLAN = {
    0: ([0], [1, 2, 3]),
    1: ([1], [0, 2, 3]),
    2: ([2], [0, 1, 3]),
    3: ([3], [0, 1, 2]),
}

# chunks per dataset for prep_data (ttH/VH/ggF_VBF are small => 4;
# bkg has the huge Stitch files => 32). Match old 4fold scheme.
DATASET_CHUNKS = {"ttH": 4, "VH": 4, "ggF_VBF": 8, "bkg": 32}
DATASET_SHUFFLE_CHUNKS = {"ttH": 4, "VH": 4, "ggF_VBF": 8, "bkg": 32}


def era_files_for(subdir):
    """Return existing era .root files (relative paths) under V3_SRC/<subdir>."""
    d = V3_SRC / subdir
    out = []
    if not d.is_dir():
        return out
    for era in ALL_ERAS:
        f = d / f"{era}.root"
        if f.exists():
            out.append(f"{subdir}/{era}.root")
    return out


def all_era_files():
    """Build {dataset_name: [relative file paths]} for the 4 datasets."""
    out = {}
    for name, label, subdirs in DATASETS:
        files = []
        for sub in subdirs:
            files.extend(era_files_for(sub))
        out[name] = files
    return out


# --------------------------------------------------------------------------
# node branch layout (identical to old pt_only 4fold config)
# --------------------------------------------------------------------------
def node_branch_names_block():
    # 6 features x 16 nodes (6 jets, 2 bjets, 6 leps, photon, MET)
    # feat0: pT (lep/photon use _div_mllg, jet/bjet/MET raw)
    feat_pt = []
    for i in range(6):
        feat_pt.append(f"jet{i}_pt")
    for i in range(2):
        feat_pt.append(f"bjet{i}_pt")
    for i in range(6):
        feat_pt.append(f"lep{i}_pt_div_mllg")
    feat_pt.append("photon_pt_div_mllg")
    feat_pt.append("MET_pt")

    feat_eta = []
    for i in range(6):
        feat_eta.append(f"jet{i}_eta")
    for i in range(2):
        feat_eta.append(f"bjet{i}_eta")
    for i in range(6):
        feat_eta.append(f"lep{i}_eta")
    feat_eta.append("photon_eta")
    feat_eta.append("0")

    feat_phi = []
    for i in range(6):
        feat_phi.append(f"jet{i}_phi")
    for i in range(2):
        feat_phi.append(f"bjet{i}_phi")
    for i in range(6):
        feat_phi.append(f"lep{i}_phi")
    feat_phi.append("photon_phi")
    feat_phi.append("MET_phi")

    feat_btag = []
    for i in range(6):
        feat_btag.append(f"jet{i}_hzg_btag_score")
    for i in range(2):
        feat_btag.append(f"bjet{i}_hzg_btag_score")
    feat_btag.extend(["0"] * 6)            # 6 leps
    # v3 upstream renamed the photon ID MVA branch from the v2 alias
    # `photon_id_mva` to the NanoAOD-standard `photon_mvaID` (same score,
    # verified bit-identical in v2 where both names existed). Use the v3 name.
    feat_btag.append("photon_mvaID")
    feat_btag.append("0")                  # MET

    feat_pdgid = ["0"] * 8                 # 6 jets + 2 bjets
    for i in range(6):
        feat_pdgid.append(f"lep{i}_pdgId")
    feat_pdgid.append("22")                # photon
    feat_pdgid.append("0")                  # MET

    node_type = "NODE_TYPE"

    return [feat_pt, feat_eta, feat_phi, feat_btag, feat_pdgid, node_type]


def yaml_list(items, indent=6):
    """Render a python list as a YAML list at the given indent."""
    lines = []
    pad = " " * indent
    if not items:
        return lines
    for it in items:
        lines.append(f"{pad}- {it}")
    return lines


def render_config(fold, files_by_ds):
    test_folds, train_folds = FOLD_PLAN[fold]
    lines = []
    a = lines.append
    a(f"Training_Name: hzg_ttH_4class_4fold_fold{fold}")
    a(f"Training_Directory: {TRAIN_DIR_BASE}_fold{fold}")
    a("Model:")
    a("  module: models.GCN")
    a("  class: Edge_Network")
    a("  args:")
    a("    hid_size: 64")
    a("    in_size: 6")
    a("    out_size: 4")
    a("    n_layers: 4")
    a("    n_proc_steps: 4")
    a("    dropout: 0")
    a("Loss:")
    a("  module: torch.nn")
    a("  class: CrossEntropyLoss")
    a("  args: {}")
    a("  finish:")
    a("    module: torch.nn")
    a("    class: Softmax")
    a("    args: {dim: 1}")
    a("Training:")
    a("  epochs: 100")
    a("  batch_size: 256")
    a("  learning_rate: 0.0001")
    a("  gamma: 0.99")
    a("Datasets:")

    node_branches = node_branch_names_block()

    first = True
    for name, label, subdirs in DATASETS:
        files = files_by_ds[name]
        chunks = DATASET_CHUNKS[name]
        shuffle_chunks = DATASET_SHUFFLE_CHUNKS[name]
        if first:
            # first dataset defines the anchors
            a(f"  {name}: &dataset_defn")
            a("    module: root_gnn_base.dataset")
            a("    class: LazyDataset")
            a(f"    shuffle_chunks: {shuffle_chunks}")
            a("    padding_mode: NONE")
            a("    args: &dataset_args")
        else:
            # subsequent datasets merge the anchor and override specific keys
            a(f"  {name}:")
            a("    <<: *dataset_defn")
            a("    args:")
            a("      <<: *dataset_args")
        a(f"      name: {name}")
        a(f"      label: {label}")
        a("      buffer_size: 2")
        a(f"      chunks: {chunks}")
        a("      file_names:")
        for f in files:
            a(f"      - {f}")
        a("      fold_var: photon_eta_fold")
        a("      node_branch_names:")
        for feat in node_branches:
            if isinstance(feat, list):
                # render as a single-line nested list for robustness
                items = ", ".join(str(b) for b in feat)
                a(f"      - [{items}]")
            else:
                a(f"      - {feat}")
        a("      node_branch_types:")
        for _ in range(16):
            a("      - single")
        a("      node_feature_scales:")
        a("      - 0.1")
        a("      - 1")
        a("      - 1")
        a("      - 1")
        a("      - 0.1")
        a("      - 1")
        a(f"      raw_dir: {V3_INPUT_DIR}")
        a(f"      save_dir: {GRAPH_DIR}")
        a("      tracking_info:")
        a("      - weight_lumi")
        a("      - source_tree_tag")
        a("      tree_name: merged_SR_Sideband")
        a("      weight_var: weight_lumi_avg")
        a("    batch_size: 256")
        a("    folding:")
        a("      n_folds: 4")
        a(f"      test: {yaml_list_inline(test_folds)}")
        a(f"      train: {yaml_list_inline(train_folds)}")
        first = False

    return "\n".join(lines) + "\n"


def yaml_list_inline(items):
    if len(items) == 1:
        return f"[{items[0]}]"
    return "[" + ", ".join(str(x) for x in items) + "]"


def render_prep_sbatch(fold):
    chunks_lines = []
    for name, _label, _sub in DATASETS:
        n = DATASET_CHUNKS[name]
        chunks_lines.append((name, n))

    lines = []
    a = lines.append
    a("#!/bin/bash")
    a("")
    a("#SBATCH --partition=gpu")
    a("#SBATCH --qos=cmsnormal")
    a("#SBATCH --account=cmsgpu")
    a(f"#SBATCH --job-name=hzg-prep-4class-fold{fold}")
    a("#SBATCH --ntasks=1")
    a("#SBATCH --cpus-per-task=4")
    a("#SBATCH --gres=gpu:1")
    a("#SBATCH --mem=32G")
    a(f"#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_4class_4fold_fold{fold}-%j.out")
    a("")
    a("set -euo pipefail")
    a("")
    a("mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs")
    a("")
    a("export PYTHONUNBUFFERED=1")
    a("export DGLBACKEND=pytorch")
    a("export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config")
    a("mkdir -p ${DGL_CONF_PATH}")
    a("")
    a(f"PYTHON={PYTHON_BIN}")
    a(f"REPO={DGL}")
    a(f"CONFIG=configs/hzg/ttH_4class_4fold_fold{fold}.yaml")
    a("")
    a("cd ${REPO}")
    a("")
    for name, n in chunks_lines:
        a(f"echo \"=== Build + shuffle graphs for {name} ({n} chunks) ===\"")
        a(f"NCHUNKS_{name.upper()}={n}")
        a(f"for chunk in $(seq 0 $((NCHUNKS_{name.upper()} - 1))); do")
        a(f"    echo \"--- {name} raw chunk ${{chunk}} ---\"")
        a(f"    ${{PYTHON}} scripts/prep_data.py \\")
        a(f"        --config ${{CONFIG}} \\")
        a(f"        --dataset {name} \\")
        a(f"        --chunk ${{chunk}}")
        a("done")
        a(f"for chunk in $(seq 0 $((NCHUNKS_{name.upper()} - 1))); do")
        a(f"    echo \"--- {name} shuffle chunk ${{chunk}} ---\"")
        a(f"    ${{PYTHON}} scripts/prep_data.py \\")
        a(f"        --config ${{CONFIG}} \\")
        a(f"        --dataset {name} \\")
        a(f"        --shuffle_mode \\")
        a(f"        --chunk ${{chunk}}")
        a("done")
        a("")
    a(f"echo \"=== Prep done for fold{fold} ===\"")
    return "\n".join(lines) + "\n"


def render_training_sbatch(fold):
    lines = []
    a = lines.append
    a("#!/bin/bash")
    a("")
    a("#SBATCH --partition=gpu")
    a("#SBATCH --qos=cmsnormal")
    a("#SBATCH --account=cmsgpu")
    a(f"#SBATCH --job-name=hzg-4class-fold{fold}")
    a("#SBATCH --ntasks=1")
    a("#SBATCH --cpus-per-task=2")
    a("#SBATCH --gres=gpu:1")
    a("#SBATCH --mem=32G")
    a(f"#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/train_4class_4fold_fold{fold}-%j.out")
    a("")
    a("set -euo pipefail")
    a("")
    a("mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs")
    a(f"mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/ttH_4class_4fold_fold{fold}")
    a("")
    a("export PYTHONUNBUFFERED=1")
    a("export DGLBACKEND=pytorch")
    a("export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config")
    a("export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib")
    a("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}")
    a("mkdir -p ${DGL_CONF_PATH} ${MPLCONFIGDIR}")
    a("")
    a(f"PYTHON={PYTHON_BIN}")
    a(f"REPO={DGL}")
    a(f"CONFIG=configs/hzg/ttH_4class_4fold_fold{fold}.yaml")
    a("")
    a("cd ${REPO}")
    a("")
    a(f"echo \"=== Training: ttH 4class 4fold fold{fold} ===\"")
    a('echo "Config: ${CONFIG}"')
    a('echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"')
    a("")
    a("srun ${PYTHON} scripts/training_script.py \\")
    a("    --config ${CONFIG} \\")
    a("    --preshuffle \\")
    a("    --nocompile \\")
    a("    --lazy \\")
    a("    --restart")
    a("")
    a('echo "=== Done ==="')
    return "\n".join(lines) + "\n"


def render_inference_sbatch(fold, files_by_ds):
    # score every input file used in training (all datasets' era files)
    all_files = []
    for name, _label, _sub in DATASETS:
        all_files.extend(files_by_ds[name])
    # dedupe preserving order (ggF_VBF dataset pulls from two subdirs)
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    lines = []
    a = lines.append
    a("#!/bin/bash")
    a("")
    a("#SBATCH --partition=gpu")
    a("#SBATCH --qos=cmsnormal")
    a("#SBATCH --account=cmsgpu")
    a(f"#SBATCH --job-name=hzg-infer-4class-fold{fold}")
    a("#SBATCH --ntasks=1")
    a("#SBATCH --cpus-per-task=4")
    a("#SBATCH --gres=gpu:1")
    a("#SBATCH --mem=32G")
    a(f"#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_4class_4fold_fold{fold}-%j.out")
    a("")
    a("set -euo pipefail")
    a("")
    a("mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs")
    a("")
    a("export PYTHONUNBUFFERED=1")
    a("export DGLBACKEND=pytorch")
    a("export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config")
    a("export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib")
    a("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}")
    a("mkdir -p ${DGL_CONF_PATH} ${MPLCONFIGDIR}")
    a("")
    a(f"PYTHON={PYTHON_BIN}")
    a(f"REPO={DGL}")
    a(f"CONFIG=configs/hzg/ttH_4class_4fold_fold{fold}.yaml")
    a("")
    a(f"INPUT_DIR={V3_INPUT_DIR}")
    a(f"OUTPUT_DIR={SCORED_DIR_BASE}_fold{fold}")
    a("")
    a("cd ${REPO}")
    a("")
    a(f"echo \"=== Inference: ttH 4class 4fold fold{fold} ===\"")
    a('echo "Config: ${CONFIG}"')
    a('echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"')
    a("")
    a(f"total={len(unique_files)}")
    a("count=0")
    a("")
    for rel in unique_files:
        a(f'REL="{rel}"')
        a('INPUT="${INPUT_DIR}/${REL}"')
        a('OUTPUT="${OUTPUT_DIR}/${REL}"')
        a('if [ ! -f "${INPUT}" ]; then')
        a('    echo "[skip] missing: ${REL}"')
        a("    continue")
        a("fi")
        a('mkdir -p "$(dirname "${OUTPUT}")"')
        a('if [ -f "${OUTPUT}" ]; then')
        a('    echo "[skip] exists: ${REL}"')
        a("    continue")
        a("fi")
        a('count=$((count + 1))')
        a('echo "[${count}/${total}] Scoring: ${REL}"')
        a("${PYTHON} scripts/inference.py \\")
        a("    --config ${CONFIG} \\")
        a('    --target "${INPUT}" \\')
        a('    --destination "${OUTPUT}" \\')
        a("    --branch_name ttH_gnn_score \\")
        a("    --chunks 1 \\")
        a("    --chunkno 0 \\")
        a("    --write \\")
        a("    --tree merged_SR_Sideband")
        a("")
    a('echo "=== Done ==="')
    return "\n".join(lines) + "\n"


def render_inference_data_sbatch(fold):
    """Score Data (all eras) for Z / sideband analysis."""
    data_files = era_files_for("Data/Data")
    lines = []
    a = lines.append
    a("#!/bin/bash")
    a("")
    a("#SBATCH --partition=gpu")
    a("#SBATCH --qos=cmsnormal")
    a("#SBATCH --account=cmsgpu")
    a(f"#SBATCH --job-name=hzg-infer-4class-data-fold{fold}")
    a("#SBATCH --ntasks=1")
    a("#SBATCH --cpus-per-task=4")
    a("#SBATCH --gres=gpu:1")
    a("#SBATCH --mem=32G")
    a(f"#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_4class_4fold_data_fold{fold}-%j.out")
    a("")
    a("set -euo pipefail")
    a("")
    a("mkdir -p /scratchfs2/cms/huangsuyun/GNN4Colliders/logs")
    a("")
    a("export PYTHONUNBUFFERED=1")
    a("export DGLBACKEND=pytorch")
    a("export DGL_CONF_PATH=/scratchfs2/cms/huangsuyun/dgl_config")
    a("export MPLCONFIGDIR=/scratchfs2/cms/huangsuyun/GNN4Colliders/cache/matplotlib")
    a("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}")
    a("mkdir -p ${DGL_CONF_PATH} ${MPLCONFIGDIR}")
    a("")
    a(f"PYTHON={PYTHON_BIN}")
    a(f"REPO={DGL}")
    a(f"CONFIG=configs/hzg/ttH_4class_4fold_fold{fold}.yaml")
    a("")
    a(f"INPUT_DIR={V3_INPUT_DIR}")
    a("# Data IS produced by prepare_pt_only_inputs_v3.py (it has weight_lumi=1")
    a("# in v3, so it goes through the same pt_only pipeline as MC).")
    a(f"OUTPUT_DIR={SCORED_DIR_BASE}_data_fold{fold}")
    a("")
    a("cd ${REPO}")
    a("")
    a(f"echo \"=== Inference: ttH 4class 4fold Data fold{fold} ===\"")
    a("")
    a(f"total={len(data_files)}")
    a("count=0")
    a("")
    for rel in data_files:
        a(f'REL="{rel}"')
        a('INPUT="${INPUT_DIR}/${REL}"')
        a('OUTPUT="${OUTPUT_DIR}/${REL}"')
        a('if [ ! -f "${INPUT}" ]; then')
        a('    echo "[skip] missing: ${REL}"')
        a("    continue")
        a("fi")
        a('mkdir -p "$(dirname "${OUTPUT}")"')
        a('if [ -f "${OUTPUT}" ]; then')
        a('    echo "[skip] exists: ${REL}"')
        a("    continue")
        a("fi")
        a('count=$((count + 1))')
        a('echo "[${count}/${total}] Scoring: ${REL}"')
        a("${PYTHON} scripts/inference.py \\")
        a("    --config ${CONFIG} \\")
        a('    --target "${INPUT}" \\')
        a('    --destination "${OUTPUT}" \\')
        a("    --branch_name ttH_gnn_score \\")
        a("    --chunks 1 \\")
        a("    --chunkno 0 \\")
        a("    --write \\")
        a("    --tree merged_SR_Sideband")
        a("")
    a('echo "=== Done ==="')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args()

    files_by_ds = all_era_files()
    print("Dataset file counts:")
    for name, _label, _sub in DATASETS:
        print(f"  {name:10s} label={_label}  files={len(files_by_ds[name])}")
    print()

    targets = []
    for fold in range(4):
        cfg = render_config(fold, files_by_ds)
        targets.append((DGL / "configs" / "hzg" / f"ttH_4class_4fold_fold{fold}.yaml", cfg))
        targets.append((DGL / "jobs" / "prep_data" / f"hzg_ttH_4class_4fold_fold{fold}_prep.sbatch", render_prep_sbatch(fold)))
        targets.append((DGL / "jobs" / "training" / f"hzg_ttH_4class_4fold_fold{fold}.sbatch", render_training_sbatch(fold)))
        targets.append((DGL / "jobs" / "inference" / f"hzg_ttH_4class_4fold_fold{fold}_inference.sbatch", render_inference_sbatch(fold, files_by_ds)))
    # one Data inference job per fold (Data read from merged dir; see note)
    for fold in range(4):
        targets.append((DGL / "jobs" / "inference" / f"hzg_ttH_4class_4fold_inference_data_fold{fold}.sbatch", render_inference_data_sbatch(fold)))

    for path, content in targets:
        if args.dry_run:
            print(f"=== {path} ===")
            print(content)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"wrote {path}")
    print(f"\n{len(targets)} files {'(dry-run, none written)' if args.dry_run else 'written'}.")


if __name__ == "__main__":
    main()
