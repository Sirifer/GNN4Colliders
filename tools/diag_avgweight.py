#!/usr/bin/env python3
"""
Diagnostic for the GNN-high category avg-weight plot.

Question: the avg-weight plot of GNN-high shows ~100 events but the
statistical fluctuations look like only ~10 events.

Hypothesis: weight_lumi_avg is normalized so that sum(w_avg) == N_events,
so per-event relative weights are identical to weight_lumi. A handful of
high-weight events then dominate both the bin content and the error
sqrt(sum(w^2)), collapsing Neff far below the raw event count.

This script reuses the loading/OOF logic of significance_diagnostic.py and
breaks the GNN-high category down per sample / per event, reporting
weight_lumi, weight_lumi_avg, Neff, and the top-weighted events.
"""

import sys
from pathlib import Path

import numpy as np
import uproot

# -----------------------------------------------------------------------------
# Config (mirror significance_diagnostic.py)
# -----------------------------------------------------------------------------

DATA_DIR_FOLD0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
)
DATA_DIR_FOLD1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold1"
)

TREE = "merged_SR_Sideband"
WEIGHT = "weight_lumi"
WEIGHT_AVG = "weight_lumi_avg"
MASS = "m_llg"
SCORE = "ttH_gnn_score"
IS_SR = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR = "photon_eta_fold"

BRANCHES = [
    WEIGHT,
    WEIGHT_AVG,
    MASS,
    SCORE,
    IS_SR,
    TREE_TAG,
    FOLDVAR,
]

FOLD0_OOF_PARITY = 0
FOLD1_OOF_PARITY = 1

MASS_LO = 120.0
MASS_HI = 130.0

VALID_SR_TAGS = np.array([0, 2, 4, 6, 8])
VALID_SB_TAGS = VALID_SR_TAGS + 1
VALID_FULL_TAGS = np.concatenate([VALID_SR_TAGS, VALID_SB_TAGS])

RUN3 = [
    "2022preEE",
    "2022postEE",
    "2023preBPix",
    "2023postBPix",
    "2024",
    "2025",
]

SIGNAL_GROUP = ("Signal/ttH", RUN3)

BACKGROUND_GROUPS = [
    ("Signal/VH", RUN3),
    ("Signal/ggF", RUN3),
    ("Signal/VBF", RUN3),
    ("Bkg_MC/TT", RUN3),
    ("Bkg_MC/DYGto2LG_10to100", RUN3),
]

# Use the same c2 as the production script's Bayesian optimum.
# Fill after first run if unknown; placeholder triggers a recompute.
C2 = 0.95

BIN_WIDTH = 4.0
BINS = np.arange(100.0, 180.0 + BIN_WIDTH, BIN_WIDTH)


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def empty_arrays():
    return {b: np.array([]) for b in BRANCHES}


def concatenate(arrays, extra_keys=None):
    keys = list(BRANCHES)
    if extra_keys:
        keys = list(set(keys) | set(extra_keys))
    out = {}
    for b in keys:
        pieces = [a[b] for a in arrays if a[b].size > 0]
        out[b] = np.concatenate(pieces) if pieces else np.array([])
    return out


def apply_mask(arr, mask):
    return {b: arr[b][mask] for b in arr}


def load_file(path):
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return empty_arrays()
            tree = f[TREE]
            missing = [b for b in BRANCHES if b not in tree.keys()]
            if missing:
                print(f"WARNING: {path} missing {missing}", file=sys.stderr)
                return empty_arrays()
            return tree.arrays(BRANCHES, library="np")
    except Exception as exc:
        print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return empty_arrays()


def load_group(data_dir, subdir, years):
    pieces = []
    for year in years:
        path = data_dir / subdir / f"{year}.root"
        if not path.exists():
            continue
        arr = load_file(path)
        if arr[WEIGHT].size > 0:
            arr["__sample__"] = np.full(arr[WEIGHT].shape, subdir, dtype=object)
            arr["__year__"] = np.full(arr[WEIGHT].shape, year, dtype=object)
            pieces.append(arr)
    return concatenate(pieces, extra_keys=["__sample__", "__year__"]) if pieces else {**empty_arrays(), "__sample__": np.array([], dtype=object), "__year__": np.array([], dtype=object)}


def oof_fullrange_mask(arr, parity):
    return (
        np.isin(arr[TREE_TAG], VALID_FULL_TAGS)
        & ((arr[FOLDVAR] % 2) == parity)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[MASS])
        & np.isfinite(arr[SCORE])
    )


def oof_sr_mask(arr, parity):
    return (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
        & ((arr[FOLDVAR] % 2) == parity)
    )


def build_oof(arr_fold0, arr_fold1, mask_builder):
    p0 = apply_mask(arr_fold0, mask_builder(arr_fold0, FOLD0_OOF_PARITY))
    p1 = apply_mask(arr_fold1, mask_builder(arr_fold1, FOLD1_OOF_PARITY))
    return concatenate([p0, p2_safe(p1)], extra_keys=["__sample__", "__year__"])


def p2_safe(a):
    return a


def build_oof_tagged(arr_fold0, arr_fold1, mask_builder):
    """OOF builder that preserves __sample__ / __year__ tags."""
    p0 = apply_mask(arr_fold0, mask_builder(arr_fold0, FOLD0_OOF_PARITY))
    p1 = apply_mask(arr_fold1, mask_builder(arr_fold1, FOLD1_OOF_PARITY))
    return concatenate([p0, p1], extra_keys=["__sample__", "__year__"])


# -----------------------------------------------------------------------------
# Optuna-free c2 (grid scan on SR, same objective as production)
# -----------------------------------------------------------------------------

def asimov_z(s, b):
    if s <= 0.0 or b <= 0.0:
        return 0.0
    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def find_best_c2(signal_oof, background_oof):
    """Grid-scan 3-category combined Z and return best c1,c2."""
    s_score = signal_oof[SCORE]
    b_score = background_oof[SCORE]
    s_w = signal_oof[WEIGHT]
    b_w = background_oof[WEIGHT]

    best = None
    best_z = -1.0
    c1_vals = np.arange(0.10, 0.951, 0.01)
    c2_vals = np.arange(0.20, 1.000, 0.005)
    for c1 in c1_vals:
        for c2 in c2_vals:
            if c2 <= c1 + 0.001:
                continue
            # 3-cat combined Z over high/medium/low
            z2 = 0.0
            ok = True
            for sm, bm in [
                (s_score > c2, b_score > c2),
                ((s_score > c1) & (s_score <= c2), (b_score > c1) & (b_score <= c2)),
                (s_score <= c1, b_score <= c1),
            ]:
                S = float(np.sum(s_w[sm]))
                B = float(np.sum(b_w[bm]))
                if B <= 0.0:
                    ok = False
                    break
                z = asimov_z(S, B)
                z2 += z * z
            if not ok:
                continue
            z = np.sqrt(z2)
            if z > best_z:
                best_z = z
                best = (float(c1), float(c2))
    return best, best_z


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def neff(w):
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return (s * s / s2) if s2 > 0.0 else 0.0


def report_sample(name, w_lumi, w_avg, mass, score, tag):
    n = w_lumi.size
    print(f"\n{'='*78}")
    print(f"{name}   N(events)={n}")
    print(f"{'='*78}")
    if n == 0:
        print("  (empty)")
        return
    sum_lumi = float(np.sum(w_lumi))
    sum_avg = float(np.sum(w_avg))
    print(f"  sum(weight_lumi)      = {sum_lumi:.6f}")
    print(f"  sum(weight_lumi_avg)  = {sum_avg:.6f}")
    print(f"  ratio avg/lumi        = {sum_avg/sum_lumi:.6e}" if sum_lumi != 0 else "")
    print(f"  Neff(weight_lumi)     = {neff(w_lumi):.4f}")
    print(f"  Neff(weight_lumi_avg) = {neff(w_avg):.4f}")
    print(f"  Neff / N              = {neff(w_avg)/n:.4f}")
    print(f"  sqrt(sum(w_lumi^2))   = {np.sqrt(np.sum(w_lumi*w_lumi)):.6f}")
    print(f"  sqrt(sum(w_avg^2))    = {np.sqrt(np.sum(w_avg*w_avg)):.6f}")
    # top events
    order = np.argsort(w_lumi)[::-1]
    print(f"\n  Top {min(15,n)} events by weight_lumi:")
    print(f"    {'i':>4} {'w_lumi':>14} {'w_avg':>12} {'m_llg':>8} {'score':>8} {'tag':>4}")
    for k, i in enumerate(order[:15]):
        print(f"    {i:>4} {w_lumi[i]:>14.6f} {w_avg[i]:>12.6f} {mass[i]:>8.3f} {score[i]:>8.4f} {int(tag[i]):>4}")


def main():
    print("Loading fold0 ...")
    signal_fold0 = load_group(DATA_DIR_FOLD0, SIGNAL_GROUP[0], SIGNAL_GROUP[1])
    bkg_parts = {name: load_group(DATA_DIR_FOLD0, name, RUN3) for name, _ in [
        ("Signal/VH", RUN3), ("Signal/ggF", RUN3), ("Signal/VBF", RUN3),
        ("Bkg_MC/TT", RUN3), ("Bkg_MC/DYGto2LG_10to100", RUN3),
    ]}

    # OOF SR for c2 search
    print("Building OOF SR for c2 search ...")
    sg0 = signal_fold0
    bg0_all = {n: bkg_parts[n] for n in bkg_parts}

    print("Loading fold1 ...")
    signal_fold1 = load_group(DATA_DIR_FOLD1, SIGNAL_GROUP[0], SIGNAL_GROUP[1])
    bkg_parts_f1 = {name: load_group(DATA_DIR_FOLD1, name, RUN3) for name, _ in [
        ("Signal/VH", RUN3), ("Signal/ggF", RUN3), ("Signal/VBF", RUN3),
        ("Bkg_MC/TT", RUN3), ("Bkg_MC/DYGto2LG_10to100", RUN3),
    ]}

    signal_oof = build_oof_tagged(signal_fold0, signal_fold1, oof_sr_mask)
    bg0_concat = concatenate(list(bg0_all.values()), extra_keys=["__sample__", "__year__"])
    bg1_concat = concatenate(list(bkg_parts_f1.values()), extra_keys=["__sample__", "__year__"])
    background_oof = build_oof_tagged(bg0_concat, bg1_concat, oof_sr_mask)

    print("\nSearching best c1/c2 (grid, SR, 3-cat combined Z) ...")
    (c1, c2), z = find_best_c2(signal_oof, background_oof)
    print(f"Best: c1={c1:.4f}, c2={c2:.4f}, combined Z={z:.4f}")

    # Now the GNN-high full-range OOF sample (what the plot uses)
    print("\nBuilding GNN-high full-range OOF sample ...")
    sig_full = build_oof_tagged(signal_fold0, signal_fold1, oof_fullrange_mask)
    bg_full = build_oof_tagged(bg0_concat, bg1_concat, oof_fullrange_mask)

    sig_high = sig_full[SCORE] > c2
    bg_high = bg_full[SCORE] > c2

    print("\n" + "#" * 78)
    print(f"# GNN-HIGH CATEGORY  (score > {c2:.6f}, full-range OOF)")
    print("#" * 78)

    report_sample(
        "SIGNAL (ttH) high",
        sig_full[WEIGHT][sig_high], sig_full[WEIGHT_AVG][sig_high],
        sig_full[MASS][sig_high], sig_full[SCORE][sig_high], sig_full[TREE_TAG][sig_high],
    )

    # Per background sample breakdown
    for name in ["Signal/VH", "Signal/ggF", "Signal/VBF", "Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100"]:
        m0 = (bg0_concat["__sample__"] == name)
        # tag arrays needed for full-range OOF; rebuild per-sample OOF full
        s0 = bg0_all[name]
        s1 = bkg_parts_f1[name]
        oof = build_oof_tagged(s0, s1, oof_fullrange_mask)
        high = oof[SCORE] > c2
        report_sample(
            f"BKG {name} high",
            oof[WEIGHT][high], oof[WEIGHT_AVG][high],
            oof[MASS][high], oof[SCORE][high], oof[TREE_TAG][high],
        )

    # Total background high
    report_sample(
        "TOTAL BKG high",
        bg_full[WEIGHT][bg_high], bg_full[WEIGHT_AVG][bg_high],
        bg_full[MASS][bg_high], bg_full[SCORE][bg_high], bg_full[TREE_TAG][bg_high],
    )


if __name__ == "__main__":
    main()
