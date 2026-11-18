#!/usr/bin/env python3
"""
Look up the event ID of the fold1 test high-category heavy background event.

Fingerprint (from the 2026-10-27 log):
    weight_lumi ~ 0.7448, ttH_gnn_score ~ 0.994, m_llg ~ 126.0,
    source_tree_tag = ttHhad (0), parity = fold1 test = 1,
    source = Bkg_MC/TT/2024.

This script scans the fold1 background files (TT/2024 first, then every other
background sample for completeness) and prints every event matching the
fingerprint, with all of its branches — including `event`, `run`,
`luminosityBlock` if present — so we get the real event ID and can confirm the
fingerprint is unique (must be exactly one match).
"""

from pathlib import Path

import numpy as np
import uproot

# --- Config (mirrors significance_4fold.py) -----------------------------------
FOLD1_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders"
    "/root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold1"
)
TREE     = "merged_SR_Sideband"
WEIGHT   = "weight_lumi"
MASS     = "m_llg"
SCORE    = "ttH_gnn_score"
IS_SR    = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR  = "photon_eta_fold"

# Fingerprint windows (loose enough to tolerate rounding in the log).
W_LO, W_HI   = 0.74, 0.75
S_LO, S_HI   = 0.993, 0.995
M_LO, M_HI   = 125.5, 126.5
TAG          = 0          # ttHhad
PARITY       = 1          # fold1 test parity

BG_SAMPLES = [
    "Bkg_MC/TT",
    "Bkg_MC/DYGto2LG_10to100",
    "Signal/VH",
    "Signal/ggF",
    "Signal/VBF",
]
YEARS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]


def scan_file(path):
    """Return list of (dict_of_branch_values) for matching events in this file."""
    with uproot.open(str(path)) as f:
        if TREE not in f:
            return [], []
        tree = f[TREE]
        present = list(tree.keys())
        # Load the fingerprint branches plus any ID-like branches that exist.
        core = [WEIGHT, MASS, SCORE, IS_SR, TREE_TAG, FOLDVAR]
        id_branches = [b for b in ["event", "run", "luminosityBlock"] if b in present]
        load = core + id_branches
        arr = tree.arrays(load, library="np")

    sr = arr[IS_SR] == 1
    finite = np.isfinite(arr[WEIGHT]) & np.isfinite(arr[SCORE]) & np.isfinite(arr[MASS])
    match = (
        sr & finite
        & (arr[WEIGHT]   > W_LO) & (arr[WEIGHT]   < W_HI)
        & (arr[SCORE]    > S_LO) & (arr[SCORE]    < S_HI)
        & (arr[MASS]     > M_LO) & (arr[MASS]     < M_HI)
        & (arr[TREE_TAG] == TAG)
        & ((arr[FOLDVAR] % 4) == PARITY)
    )
    idx = np.where(match)[0]
    return [arr], list(idx), id_branches


def main():
    total_matches = 0
    print("=" * 78)
    print("Looking up fold1 test high heavy background event")
    print(f"Fingerprint: weight∈({W_LO},{W_HI}) score∈({S_LO},{S_HI}) "
          f"mass∈({M_LO},{M_HI}) tag={TAG} parity={PARITY}")
    print("=" * 78)

    for sample in BG_SAMPLES:
        for year in YEARS:
            path = FOLD1_DIR / sample / f"{year}.root"
            if not path.exists():
                continue
            try:
                arr, idx, id_branches = scan_file(path)
            except Exception as exc:
                print(f"  [skip] {sample}/{year}: {exc}")
                continue
            if not idx:
                continue
            arr = arr[0]
            print(f"\n>>> {sample}/{year}: {len(idx)} matching event(s)")
            for i in idx:
                total_matches += 1
                print(f"    index      = {int(i)}")
                print(f"    weight_lumi= {float(arr[WEIGHT][i]):.6f}")
                print(f"    score      = {float(arr[SCORE][i]):.6f}")
                print(f"    m_llg      = {float(arr[MASS][i]):.4f}")
                print(f"    tree_tag   = {int(arr[TREE_TAG][i])}  (parity={int(arr[FOLDVAR][i] % 4)})")
                print(f"    is_SR      = {int(arr[IS_SR][i])}")
                print(f"    photon_eta_fold = {int(arr[FOLDVAR][i])}")
                for b in id_branches:
                    v = arr[b][i]
                    try:
                        print(f"    {b:<14}= {int(v)}")
                    except (TypeError, ValueError):
                        print(f"    {b:<14}= {v}")
            print(f"    (branches present in this file: {sorted(arr.keys())})")

    print("\n" + "=" * 78)
    if total_matches == 1:
        print(f"OK: exactly 1 match across all background samples — fingerprint is unique.")
    elif total_matches == 0:
        print("WARNING: no matches — fingerprint windows too tight? check values.")
    else:
        print(f"WARNING: {total_matches} matches — fingerprint is NOT unique; "
              "narrow the windows before trusting a drop.")
    print("=" * 78)


if __name__ == "__main__":
    main()
