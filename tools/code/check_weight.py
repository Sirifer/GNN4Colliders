#!/usr/bin/env python3

import uproot
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_merged/Signal/ttH"
)

# 需要检查的 ROOT 文件
FILES = [
    "2022preEE.root",
    "2022postEE.root",
    "2023preBPix.root",
    "2023postBPix.root",
    "2024.root",
    "2025.root",

]

# 需要检查的 trees
TREES = [
    # "SR__ttHhad",
    # "Sideband__ttHhad",
    # "SR__ttHlep",
    # "Sideband__ttHlep",
    # "SR__untag",
    # "Sideband__untag",
    # "SR__VH3l",
    # "Sideband__VH3l",
    # "SR__VHpTmiss",
    # "Sideband__VHpTmiss",
    # "SR__inclusive",
    # "Sideband__inclusive",
    "merged_SR_Sideband",
]

BRANCH = "weight_lumi"

# ============================================================
# Main
# ============================================================

grand_total = 0.0

print(f"{'File':<22} {'Tree':<25} {'Entries':>12} {'Sum(weight_lumi)':>20}")
print("-" * 85)

for filename in FILES:
    filepath = ROOT_DIR / filename

    if not filepath.exists():
        print(f"[WARNING] File not found: {filepath}")
        continue

    with uproot.open(filepath) as f:

        for tree_name in TREES:

            if tree_name not in f:
                print(f"{filename:<22} {tree_name:<25} {'TREE NOT FOUND':>32}")
                continue

            tree = f[tree_name]

            if BRANCH not in tree.keys():
                print(f"{filename:<22} {tree_name:<25} {'BRANCH NOT FOUND':>32}")
                continue

            weights = tree[BRANCH].array(library="np")

            tree_sum = np.sum(weights)
            n_entries = len(weights)

            grand_total += tree_sum

            print(
                f"{filename:<22} "
                f"{tree_name:<25} "
                f"{n_entries:>12d} "
                f"{tree_sum:>20.6f}"
            )

print("-" * 85)
print(f"{'TOTAL':<60} {grand_total:>20.6f}")