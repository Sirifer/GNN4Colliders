#!/usr/bin/env python3
"""
Diagnose which events drive the fold0/fold1 yield imbalance
in background ttHhad (tag=0) and ttHlep (tag=2).

For each (tree, fold, sample) combination:
  - print the top-N events by |weight|
  - print cumulative fraction of total yield they represent
"""

from pathlib import Path
import sys
import numpy as np
import uproot

# =============================================================================
# Config
# =============================================================================

FOLD0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_scored_ttH_vs_other_run3v2_fold0_0828"
)
FOLD1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_scored_ttH_vs_other_run3v2_fold1_0828"
)

TREE = "merged_SR_Sideband"
WEIGHT = "weight_lumi"
MASS   = "m_llg"
IS_SR  = "is_SR"
TAG    = "source_tree_tag"
EVENT  = "event"
PHI    = "photon_eta_fold"

# Try to load run/lumi if present (not required)
OPTIONAL = ["run", "lumi"]

REQUIRED = [WEIGHT, MASS, IS_SR, TAG, EVENT, PHI]

MASS_LO, MASS_HI = 120.0, 130.0
VALID_TAGS = np.array([0, 2, 4, 6, 8])

TREE_MAP = {0: "ttHhad", 2: "ttHlep", 4: "untag", 6: "VH3l", 8: "VHpTmiss"}

RUN3 = [
    "2022preEE", "2022postEE", "2023preBPix",
    "2023postBPix", "2024", "2025",
]

BACKGROUND = [
    ("Signal/VH",                RUN3),
    ("Signal/ggF",               RUN3),
    ("Signal/VBF",               RUN3),
    ("Bkg_MC/TT",                RUN3),
    ("Bkg_MC/DYGto2LG_10to100", RUN3),
]

# Which trees to diagnose
DIAG_TAGS = [0, 2]   # ttHhad, ttHlep

TOP_N = 20   # show top-N events per (sample, fold, tree)

# =============================================================================
# Load helpers
# =============================================================================

def load_file(path):
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return None
            tree = f[TREE]
            keys = set(tree.keys())
            missing = [b for b in REQUIRED if b not in keys]
            if missing:
                print(f"  SKIP {path.name}: missing {missing}", file=sys.stderr)
                return None
            extra = [b for b in OPTIONAL if b in keys]
            branches = REQUIRED + extra
            arr = tree.arrays(branches, library="np")
            arr["_extra"] = extra   # tag which optional cols exist
            return arr
    except Exception as e:
        print(f"  WARNING: {path}: {e}", file=sys.stderr)
        return None


def physics_mask(arr, parity):
    return (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TAG], VALID_TAGS)
        & (arr[PHI] % 2 == parity)
        & np.isfinite(arr[WEIGHT])
    )


def load_sample_by_tag(data_dir, subdir, years, parity, diag_tags):
    """
    Returns dict: tag -> list of (weight, event, year, extra_cols_dict)
    """
    out = {t: [] for t in diag_tags}

    for year in years:
        path = data_dir / subdir / f"{year}.root"
        if not path.exists():
            continue

        arr = load_file(path)
        if arr is None:
            continue

        extra_cols = arr["_extra"]
        mask = physics_mask(arr, parity)

        for tag in diag_tags:
            sel = mask & (arr[TAG] == tag)
            if not np.any(sel):
                continue

            ws  = arr[WEIGHT][sel]
            evs = arr[EVENT][sel]
            extra = {c: arr[c][sel] for c in extra_cols}

            for i in range(ws.size):
                row = {
                    "weight": float(ws[i]),
                    "event":  int(evs[i]),
                    "year":   year,
                }
                for c in extra_cols:
                    row[c] = int(extra[c][i])
                out[tag].append(row)

    return out

# =============================================================================
# Reporting
# =============================================================================

def report_top(rows, tag, fold_name, sample, top_n):
    tree_name = TREE_MAP[tag]

    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: abs(r["weight"]), reverse=True)
    total_y   = sum(r["weight"] for r in rows_sorted)
    total_abs = sum(abs(r["weight"]) for r in rows_sorted)

    if abs(total_y) < 1e-12:
        return

    has_run  = "run"  in rows_sorted[0]
    has_lumi = "lumi" in rows_sorted[0]

    print(
        f"\n  [{fold_name}] {sample} / {tree_name}  "
        f"N={len(rows_sorted)}  Y={total_y:.6f}"
    )

    col_w = 10
    hdr = f"  {'rank':>5}  {'weight':>12}  {'cum%':>7}  {'event':>12}"
    if has_run:
        hdr += f"  {'run':>10}"
    if has_lumi:
        hdr += f"  {'lumi':>8}"
    hdr += f"  {'year'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    cum = 0.0
    for i, r in enumerate(rows_sorted[:top_n]):
        cum += abs(r["weight"])
        line = (
            f"  {i+1:>5}  {r['weight']:>12.6f}  "
            f"{100*cum/total_abs:>6.1f}%  {r['event']:>12}"
        )
        if has_run:
            line += f"  {r['run']:>10}"
        if has_lumi:
            line += f"  {r['lumi']:>8}"
        line += f"  {r['year']}"
        print(line)


def compare_folds(rows0, rows1, tag, sample):
    """Show fold0 vs fold1 yield for this (sample, tree)."""
    tree_name = TREE_MAP[tag]
    y0 = sum(r["weight"] for r in rows0)
    y1 = sum(r["weight"] for r in rows1)
    rat = y1 / y0 if abs(y0) > 1e-12 else float("nan")
    print(
        f"  {sample:<34}  {tree_name:<9}  "
        f"Y0={y0:>10.5f}  Y1={y1:>10.5f}  Y1/Y0={rat:>7.3f}"
        f"  N0={len(rows0):>5}  N1={len(rows1):>5}"
    )

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("FOLD IMBALANCE DIAGNOSTIC")
    print(f"Trees: {[TREE_MAP[t] for t in DIAG_TAGS]}")
    print(f"Top-N: {TOP_N} events per (sample, fold, tree)")
    print("=" * 80)

    # ── Per-sample summary table ───────────────────────────────────────────
    print("\n" + "-" * 80)
    print("YIELD SUMMARY BY SAMPLE")
    print(f"  {'Sample':<34}  {'Tree':<9}  {'Y0':>10}  {'Y1':>10}  {'Y1/Y0':>7}  {'N0':>5}  {'N1':>5}")
    print("-" * 80)

    # store for per-sample detail below
    all_data = {}   # (subdir, tag) -> (rows0, rows1)

    for subdir, years in BACKGROUND:
        d0 = load_sample_by_tag(FOLD0, subdir, years, parity=0, diag_tags=DIAG_TAGS)
        d1 = load_sample_by_tag(FOLD1, subdir, years, parity=1, diag_tags=DIAG_TAGS)

        for tag in DIAG_TAGS:
            all_data[(subdir, tag)] = (d0[tag], d1[tag])
            compare_folds(d0[tag], d1[tag], tag, subdir)

    # ── Per-sample top-event detail ────────────────────────────────────────
    for tag in DIAG_TAGS:
        tree_name = TREE_MAP[tag]
        print("\n" + "=" * 80)
        print(f"TOP-{TOP_N} EVENTS BY |WEIGHT| — {tree_name} (tag={tag})")
        print("=" * 80)

        for subdir, years in BACKGROUND:
            rows0, rows1 = all_data[(subdir, tag)]

            y0 = sum(r["weight"] for r in rows0)
            y1 = sum(r["weight"] for r in rows1)

            # Only show samples with non-negligible yield or notable imbalance
            if abs(y0) < 1e-9 and abs(y1) < 1e-9:
                continue

            rat = abs(y1 / y0) if abs(y0) > 1e-12 else float("inf")
            notable = (rat < 0.8 or rat > 1.25) or (abs(y0) + abs(y1)) > 0.1

            if not notable:
                continue

            print(f"\n{'─'*60}")
            print(f"Sample: {subdir}")
            print(f"{'─'*60}")
            report_top(rows0, tag, "fold0", subdir, TOP_N)
            report_top(rows1, tag, "fold1", subdir, TOP_N)

    print("\nDone.")


if __name__ == "__main__":
    main()
