#!/usr/bin/env python3
"""
gen_cr_lowIDMVA_configs_jobs.py

Generate the CR_lowIDMVA inference configs and SLURM job scripts for the v3
method-2 background estimate. Produces, under root_gnn_dgl/:

  configs/hzg/cr_lowIDMVA_ttH_4class_4fold_fold{0,1,2,3}.yaml
  jobs/inference/hzg_cr_lowIDMVA_ttH_4class_4fold_inference_fold{0,1,2,3}.sbatch

These configs REUSE the already-trained fold{0-3} models -- no retraining, no
MC reprocessing. They exist only so inference.py can read/write the CR tree.

Why a separate config is required (not just --tree on the nominal config):
  inference.py's write path (scripts/inference.py:330) re-derives the tree
  name from the config via select_dataset_config -> write_config['args']
  ['tree_name'], which is the ORIGINAL config value (merged_SR_Sideband) and
  is NOT affected by the --tree CLI override (that override only mutates the
  read-side dataset dict at line 243). So the write tree is pinned to the
  config's tree_name. A CR file has no merged_SR_Sideband tree, hence the
  nominal config would fail on the write step. The CR config sets
  tree_name: merged_CR_lowIDMVA so both read and write use the CR tree.

What the CR config shares with the nominal fold config:
  - Model block (Edge_Network, in_size=6, out_size=4) -- identical, so the
    loaded checkpoint's architecture matches.
  - Training_Directory -> points at the SAME trained fold dir as nominal, so
    inference finds the checkpoint.
  - node_branch_names / node_branch_types / node_feature_scales -- identical
    6x16 layout, so the model sees the same features.
  - fold_var: photon_eta_fold (merge_cr_lowIDMVA_v3.py stamps it, same
    int32(eta*10000) encoding).

What differs:
  - tree_name: merged_CR_lowIDMVA (not merged_SR_Sideband)
  - raw_dir: the CR pt_only dir
  - One dataset "cr_data" with label 0 (label is irrelevant for inference --
    inference does not train -- but must be a valid int; the config format
    requires it). file_names lists all 11 Data era files.
  - weight_var: weight_lumi_avg (pt_only script computes it = 1.0 for Data).
  - folding block kept for schema compatibility; unused at inference time.

The node_branch_names block is duplicated here (rather than imported) to keep
this generator standalone, matching gen_4class_4fold_configs_jobs.py.

Usage:
    python gen_cr_lowIDMVA_configs_jobs.py            # write all files
    python gen_cr_lowIDMVA_configs_jobs.py --dry-run  # print, write nothing
"""

import argparse
from pathlib import Path

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
DGL = REPO / "root_gnn_dgl"

# CR pt_only input (produced by prepare_pt_only_inputs_v3.py on the CR merged dir)
CR_INPUT_DIR = REPO / "root_p2root_v3_cr_pt_only"
# CR scored output base
CR_SCORED_BASE = REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold"
# Reuse the SAME trained fold dirs as nominal (checkpoint lives here)
TRAIN_DIR_BASE = REPO / "trainings" / "hzg" / "ttH_4class_4fold_fold"

PYTHON_BIN = "/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python"

# eras present in v3 Data (Run2+Run3)
ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

CR_TREE = "merged_CR_lowIDMVA"

# 4-fold rotation kept for schema parity with nominal configs (test/train).
# Inference does not use it, but the dataset builder reads the folding key.
FOLD_PLAN = {
    0: ([0], [1, 2, 3]),
    1: ([1], [0, 2, 3]),
    2: ([2], [0, 1, 3]),
    3: ([3], [0, 1, 2]),
}


def cr_data_files():
    """Data era files relative to CR_INPUT_DIR (Data/Data/<era>.root)."""
    return [f"Data/Data/{era}.root" for era in ALL_ERAS]


def node_branch_names_block():
    # Identical 6 features x 16 nodes as nominal (see gen_4class_4fold_configs_jobs).
    feat_pt = (
        [f"jet{i}_pt" for i in range(6)]
        + [f"bjet{i}_pt" for i in range(2)]
        + [f"lep{i}_pt_div_mllg" for i in range(6)]
        + ["photon_pt_div_mllg", "MET_pt"]
    )
    feat_eta = (
        [f"jet{i}_eta" for i in range(6)]
        + [f"bjet{i}_eta" for i in range(2)]
        + [f"lep{i}_eta" for i in range(6)]
        + ["photon_eta", "0"]
    )
    feat_phi = (
        [f"jet{i}_phi" for i in range(6)]
        + [f"bjet{i}_phi" for i in range(2)]
        + [f"lep{i}_phi" for i in range(6)]
        + ["photon_phi", "MET_phi"]
    )
    feat_btag = (
        [f"jet{i}_hzg_btag_score" for i in range(6)]
        + [f"bjet{i}_hzg_btag_score" for i in range(2)]
        + ["0"] * 6
        + ["photon_mvaID", "0"]
    )
    feat_pdgid = ["0"] * 8 + [f"lep{i}_pdgId" for i in range(6)] + ["22", "0"]
    return [feat_pt, feat_eta, feat_phi, feat_btag, feat_pdgid, "NODE_TYPE"]


def yaml_list_inline(items):
    if len(items) == 1:
        return f"[{items[0]}]"
    return "[" + ", ".join(str(x) for x in items) + "]"


def render_config(fold):
    """One config per fold. Single dataset 'cr_data', label 0 (inert)."""
    files = cr_data_files()
    node_branches = node_branch_names_block()
    lines = []
    a = lines.append
    a(f"Training_Name: cr_lowIDMVA_ttH_4class_4fold_fold{fold}")
    # Reuse the SAME trained directory as nominal fold{fold} so inference
    # finds the checkpoint produced by the nominal training job.
    a(f"Training_Directory: {TRAIN_DIR_BASE}{fold}")
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
    a("  cr_data: &dataset_defn")
    a("    module: root_gnn_base.dataset")
    a("    class: LazyDataset")
    a("    shuffle_chunks: 4")
    a("    padding_mode: NONE")
    a("    args: &dataset_args")
    a("      name: cr_data")
    a("      label: 0")
    a("      buffer_size: 2")
    a("      chunks: 4")
    a("      file_names:")
    for f in files:
        a(f"      - {f}")
    a("      fold_var: photon_eta_fold")
    a("      node_branch_names:")
    for feat in node_branches:
        if isinstance(feat, list):
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
    a(f"      raw_dir: {CR_INPUT_DIR}")
    a(f"      save_dir: {REPO / 'graphs' / 'hzg' / 'cr_lowIDMVA_4class_4fold'}")
    a("      tracking_info:")
    a("      - weight_lumi")
    a("      - source_tree_tag")
    a(f"      tree_name: {CR_TREE}")
    a("      weight_var: weight_lumi_avg")
    a("    batch_size: 256")
    a("    folding:")
    a("      n_folds: 4")
    a(f"      test: {yaml_list_inline(FOLD_PLAN[fold][0])}")
    a(f"      train: {yaml_list_inline(FOLD_PLAN[fold][1])}")
    return "\n".join(lines) + "\n"


def render_inference_sbatch(fold):
    files = cr_data_files()
    lines = []
    a = lines.append
    a("#!/bin/bash")
    a("")
    a("#SBATCH --partition=gpu")
    a("#SBATCH --qos=cmsnormal")
    a("#SBATCH --account=cmsgpu")
    a(f"#SBATCH --job-name=hzg-infer-cr-fold{fold}")
    a("#SBATCH --ntasks=1")
    a("#SBATCH --cpus-per-task=4")
    a("#SBATCH --gres=gpu:1")
    a("#SBATCH --mem=32G")
    a(f"#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/infer_cr_lowIDMVA_4fold_fold{fold}-%j.out")
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
    a(f"CONFIG=configs/hzg/cr_lowIDMVA_ttH_4class_4fold_fold{fold}.yaml")
    a("")
    a(f"INPUT_DIR={CR_INPUT_DIR}/Data/Data")
    a(f"OUTPUT_DIR={CR_SCORED_BASE}_fold{fold}/Data/Data")
    a("")
    a("cd ${REPO}")
    a("")
    a(f"echo \"=== Inference: CR_lowIDMVA ttH 4class 4fold fold{fold} ===\"")
    a('echo "Config: ${CONFIG}"')
    a('echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"')
    a("")
    a(f"total={len(files)}")
    a("count=0")
    a("")
    for rel in files:
        era = rel.split("/")[-1]
        a(f'REL="{era}"')
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
        a(f"    --tree {CR_TREE}")
        a("")
    a('echo "=== Done ==="')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args()

    print(f"CR Data era files: {len(cr_data_files())}")
    print()

    targets = []
    for fold in range(4):
        cfg = render_config(fold)
        targets.append(
            (DGL / "configs" / "hzg" / f"cr_lowIDMVA_ttH_4class_4fold_fold{fold}.yaml", cfg)
        )
        targets.append(
            (
                DGL / "jobs" / "inference" / f"hzg_cr_lowIDMVA_ttH_4class_4fold_inference_fold{fold}.sbatch",
                render_inference_sbatch(fold),
            )
        )

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
