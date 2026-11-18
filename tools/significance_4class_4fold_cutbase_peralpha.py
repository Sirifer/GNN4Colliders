#!/usr/bin/env python3
"""
significance_4class_4fold_cutbased.py

Cut-based (source_tree_tag) 5-category, 4-fold OOF significance for the
v3 (Mll80to100) HZgamma production used by the 4-class GNN study.

This is the twin of `significance_4fold_cutbase.py`. Both scripts read the
same scored directories, use the same SR/SB definition (the `is_SR` branch and
`source_tree_tag`, no mass window), the same test-parity folds and the same
Asimov statistic. **The only difference is the sideband scale factor:**

    significance_4fold_cutbase.py          ONE common alpha per fold
    significance_4class_4fold_cutbased.py  ONE alpha per category

Running this script with `--common-alpha` therefore reproduces
`significance_4fold_cutbase.py` exactly (same Z, same sigma).

Why 5 categories and not 4
--------------------------
The original analysis assigns every event to one of five signal regions via
``source_tree_tag``:

    tag 0 -> ttHhad      tag 2 -> ttHlep      tag 4 -> untag
    tag 6 -> VH3l        tag 8 -> VHpTmiss

The sideband counterpart of an SR category is ``tag + 1`` (odd tags).
The 4-class GNN scheme (Other / ttH / VH / ggFVBF) is defined by softmax
argmax, and its ``Other`` class has no cut-based counterpart: cut-based
events are never "unassigned".  This script therefore reports the five
original cut-based categories; merge them externally if a 4-bin comparison
is needed (e.g. ttHhad+ttHlep, VH3l+VHpTmiss, untag).

SR / SB definition (no mass window)
-----------------------------------
    is_SR == 1  -> signal region     is_SR == 0  -> sideband
    SR tag i uses sideband tag i+1.
No mass window is applied: the tag already carries the mass selection (the
SR/`is_SR` flag is defined on the refitted mass), and `m_llg` cuts only a
subset of those events (parity-0 Data: 815 SR events with `is_SR`, 714 with
`120 < m_llg < 130`). Keeping `is_SR` is what makes the event set identical
to `significance_4fold_cutbase.py`.

Background estimate (per category, data-driven)
-----------------------------------------------
    alpha_i = N_Data_SB,i / Y_BkgMC_SB,i      (Data counts, Bkg MC weighted)
    B_i     = alpha_i * Y_BkgMC_SR,i

Data is counted, not weighted; the MC side uses `--weight-branch`. With
``--common-alpha`` a single alpha is built from the five SB tags together,
which is exactly the normalisation of `significance_4fold_cutbase.py`.

Folds
-----
    fold k: test parity = k.   Validation is NOT used (no cut is tuned here).
Each event uses the fold directory whose model did not train on its parity.
For a cut-based study no score is needed, but the same scored directories are
used so the event selections match the GNN and `significance_4fold_cutbase.py`.

Statistics
----------
    Z_i     = Asimov discovery significance of category i
    Z_split = sqrt(sum_i Z_i^2)
    Z_final = sqrt(sum_k Z_test,k^2)      (test parities are disjoint)
Uncertainties are error propagation, not a profile-likelihood fit. sigma_B
carries the alpha uncertainty per category, and the categories are combined in
quadrature; the same treatment is used in `significance_4fold_cutbase.py`, so
the two scripts differ only through alpha.

Usage
-----
    python significance_4class_4fold_cutbased.py
    python significance_4class_4fold_cutbased.py --common-alpha
    python significance_4class_4fold_cutbased.py --fold 0 --diagnostic-only
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import uproot


# =============================================================================
# 1. Configuration
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_4class_4fold_cutbased"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"

TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}

TREE_NOM = "merged_SR_Sideband"
IS_SR = "is_SR"
FOLDVAR = "photon_eta_fold"
TREE_TAG = "source_tree_tag"

# Same default as significance_4class_v3_bkg_4cat.py (Step2 convention).
DEFAULT_WEIGHT_BRANCH = "weight_lumi"

# No mass window is applied. The `is_SR` branch (defined on the refitted mass)
# is the SR/SB selector, exactly as in significance_4fold_cutbase.py; a
# `m_llg` window would select a different, smaller event set.

# Cut-based categories: (name, SR tag). SB tag = SR tag + 1.
CAT_TAGS = [
    ("ttHhad", 0),
    ("ttHlep", 2),
    ("untag", 4),
    ("VH3l", 6),
    ("VHpTmiss", 8),
]
CATS = [name for name, _ in CAT_TAGS]

SIGNAL_DIRS = [
    "Signal/ttH",
    "Signal/VH",
    "Signal/ggF",
    "Signal/VBF",
]

BKG_DIRS = [
    "Bkg_MC/TT",
    "Bkg_MC/DYGto2LG_10to100",
    "Bkg_MC/DYGto2LG_100to200",
    "Bkg_MC/DYGto2LG_200to400",
    "Bkg_MC/DYGto2LG_400to600",
    "Bkg_MC/DYGto2LG_600",
    # "Bkg_MC/DYJetsToLL_Stitch",
]

ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]


# =============================================================================
# 2. Statistics
# =============================================================================

def asimov_z(s, b):
    """Discovery Asimov significance for a single category."""
    if not np.isfinite(s) or not np.isfinite(b) or s <= 0.0 or b <= 0.0:
        return 0.0
    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def yield_info(weights):
    """Weighted yield and MC statistical uncertainty."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return {
            "sum": 0.0, "err": 0.0, "sumw2": 0.0,
            "n": 0, "nneg": 0, "neff": 0.0,
        }
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite MC weights found; refusing to calculate yields.")

    sumw = float(np.sum(w))
    sumw2 = float(np.sum(w * w))
    neff = (sumw * sumw / sumw2) if sumw2 > 0.0 else 0.0
    return {
        "sum": sumw,
        "err": float(np.sqrt(sumw2)),
        "sumw2": sumw2,
        "n": int(w.size),
        "nneg": int(np.sum(w < 0.0)),
        "neff": float(neff),
    }


def alpha_and_error(n_data, y_bkg_sb):
    """
    alpha = N_data / Y_sb with independent Poisson Data and finite-MC errors.

    Returns (alpha, sigma_alpha); NaN when the estimate is undefined.
    """
    if n_data <= 0 or y_bkg_sb["sum"] <= 0.0:
        return float("nan"), float("nan")

    alpha = n_data / y_bkg_sb["sum"]
    rel_var = (
        1.0 / n_data
        + (y_bkg_sb["err"] / y_bkg_sb["sum"]) ** 2
    )
    return float(alpha), float(alpha * np.sqrt(rel_var))


# =============================================================================
# 3. ROOT loading helpers
# =============================================================================

class EventSet:
    __slots__ = ("is_sr", "weight", "tag", "foldvar")

    def __init__(self, is_sr, weight, tag, foldvar):
        self.is_sr = np.asarray(is_sr, dtype=np.int32)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.tag = np.asarray(tag, dtype=np.int32)
        self.foldvar = np.asarray(foldvar, dtype=np.int32)

        n = self.is_sr.size
        if self.weight.size != n or self.tag.size != n or self.foldvar.size != n:
            raise ValueError("EventSet arrays have inconsistent lengths")

    @property
    def n(self):
        return self.is_sr.size


def empty_set():
    return EventSet(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )


def _concat_eventsets(parts):
    parts = [p for p in parts if p.n > 0]
    if not parts:
        return empty_set()
    return EventSet(
        np.concatenate([p.is_sr for p in parts]),
        np.concatenate([p.weight for p in parts]),
        np.concatenate([p.tag for p in parts]),
        np.concatenate([p.foldvar for p in parts]),
    )


def _load_one_dir(base, subdir, weight_branch, with_weight=True):
    """
    Load one process from one scored fold directory.

    Missing era files are reported and skipped because not every process has
    all 11 eras. An existing but unreadable file or a missing required branch
    raises an error instead of silently dropping that file.
    """
    process_dir = base / subdir
    if not process_dir.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {process_dir}")

    branches = [IS_SR, FOLDVAR, TREE_TAG]
    if with_weight:
        branches.append(weight_branch)

    pieces = []
    loaded_eras = []
    missing_eras = []

    for era in ALL_ERAS:
        path = process_dir / f"{era}.root"
        if not path.exists():
            missing_eras.append(era)
            continue

        try:
            with uproot.open(str(path)) as root_file:
                if TREE_NOM not in root_file:
                    raise KeyError(f"TTree {TREE_NOM!r} not found")
                tree = root_file[TREE_NOM]
                available = set(tree.keys())
                missing_branches = [b for b in branches if b not in available]
                if missing_branches:
                    raise KeyError(
                        f"Missing branches {missing_branches} in {path}"
                    )
                arrays = tree.arrays(branches, library="np")
        except Exception as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        is_sr = np.asarray(arrays[IS_SR], dtype=np.int32)
        foldvar = np.asarray(arrays[FOLDVAR], dtype=np.int32)
        tag = np.asarray(arrays[TREE_TAG], dtype=np.int32)

        if with_weight:
            weight = np.asarray(arrays[weight_branch], dtype=np.float64)
        else:
            # Data is counted, not weighted.
            weight = np.ones(is_sr.shape, dtype=np.float64)

        pieces.append(EventSet(is_sr, weight, tag, foldvar))
        loaded_eras.append(era)

    if not pieces:
        raise RuntimeError(
            f"No ROOT files were loaded for {subdir} under {base}"
        )

    result = _concat_eventsets(pieces)
    info = {
        "files": len(loaded_eras),
        "eras": loaded_eras,
        "missing_eras": missing_eras,
    }
    return result, info


def load_signal_all(fold, weight_branch):
    """Load and merge ttH + VH + ggF + VBF Signal MC."""
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts = []
    infos = {}

    for subdir in SIGNAL_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch)
        parts.append(eventset)
        infos[subdir] = info

    return _concat_eventsets(parts), infos


def load_background_all(fold, weight_branch):
    """Load and merge all seven background MC processes."""
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts = []
    process_sets = {}
    infos = {}

    for subdir in BKG_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch)
        process_sets[subdir] = eventset
        parts.append(eventset)
        infos[subdir] = info

    return _concat_eventsets(parts), process_sets, infos


def load_data_all(fold):
    """Load nominal Data (SR + sideband) for one fold; unweighted."""
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    return _load_one_dir(base, "Data/Data", DEFAULT_WEIGHT_BRANCH,
                         with_weight=False)


def select_parity(eventset, parity):
    mask = (eventset.foldvar % 4) == parity
    return EventSet(
        eventset.is_sr[mask],
        eventset.weight[mask],
        eventset.tag[mask],
        eventset.foldvar[mask],
    )


# =============================================================================
# 4. Normalisation and per-category evaluation
# =============================================================================

def _sb_selection(eventset, sb_tag):
    """Sideband: is_SR == 0 and the odd tag matching this category."""
    return (eventset.is_sr == 0) & (eventset.tag == sb_tag)


def _sr_selection(eventset, sr_tag):
    """Signal region: is_SR == 1 and the even tag of this category."""
    return (eventset.is_sr == 1) & (eventset.tag == sr_tag)


def compute_normalization(data_all, bkg, common_alpha=False):
    """
    Per-category alpha_i = N_Data_SB,i / Y_BkgMC_SB,i.

    With common_alpha=True a single alpha is built from the sum over all five
    SB tags, matching significance_4class_v3_bkg_4cat.py.
    """
    per_cat = {}
    tot_n_data = 0
    tot_y_sb = {"sum": 0.0, "err": 0.0, "sumw2": 0.0,
                "n": 0, "nneg": 0, "neff": 0.0}

    for name, sr_tag in CAT_TAGS:
        sb_tag = sr_tag + 1

        data_sel = _sb_selection(data_all, sb_tag)
        n_data = int(np.sum(data_sel))

        bkg_sel = _sb_selection(bkg, sb_tag) & np.isfinite(bkg.weight)
        y_sb = yield_info(bkg.weight[bkg_sel])

        alpha, sigma_alpha = alpha_and_error(n_data, y_sb)

        per_cat[name] = {
            "sr_tag": sr_tag,
            "sb_tag": sb_tag,
            "N_data_sb": n_data,
            "N_bkg_sb": y_sb["n"],
            "Y_bkg_sb": y_sb["sum"],
            "sigma_Y_bkg_sb": y_sb["err"],
            "neff_bkg_sb": y_sb["neff"],
            "nneg_bkg_sb": y_sb["nneg"],
            "alpha": alpha,
            "sigma_alpha": sigma_alpha,
            "valid": bool(np.isfinite(alpha) and alpha > 0.0),
        }

        tot_n_data += n_data
        tot_y_sb["sum"] += y_sb["sum"]
        tot_y_sb["err"] = float(np.hypot(tot_y_sb["err"], y_sb["err"]))
        tot_y_sb["sumw2"] += y_sb["sumw2"]
        tot_y_sb["n"] += y_sb["n"]

    alpha_c, sigma_alpha_c = alpha_and_error(tot_n_data, tot_y_sb)

    return {
        "per_cat": per_cat,
        "common": {
            "N_data_sb": tot_n_data,
            "N_bkg_sb": tot_y_sb["n"],
            "Y_bkg_sb": tot_y_sb["sum"],
            "sigma_Y_bkg_sb": tot_y_sb["err"],
            "alpha": alpha_c,
            "sigma_alpha": sigma_alpha_c,
            "valid": bool(np.isfinite(alpha_c) and alpha_c > 0.0),
        },
    }


def evaluate_categories(sig, bkg, norm, common_alpha=False):
    """Compute S_i, B_i and Asimov Z_i for the five cut-based categories."""
    results = {}

    for name, sr_tag in CAT_TAGS:
        sig_sel = _sr_selection(sig, sr_tag) & np.isfinite(sig.weight)
        bkg_sel = _sr_selection(bkg, sr_tag) & np.isfinite(bkg.weight)

        s_info = yield_info(sig.weight[sig_sel])
        y_info = yield_info(bkg.weight[bkg_sel])

        if common_alpha:
            alpha = norm["common"]["alpha"]
            sigma_alpha = norm["common"]["sigma_alpha"]
            valid_alpha = norm["common"]["valid"]
        else:
            cat_norm = norm["per_cat"][name]
            alpha = cat_norm["alpha"]
            sigma_alpha = cat_norm["sigma_alpha"]
            valid_alpha = cat_norm["valid"]

        y_bkg_sr = y_info["sum"]
        b_value = alpha * y_bkg_sr
        sigma_b = float(np.hypot(
            y_bkg_sr * sigma_alpha,
            alpha * y_info["err"],
        ))

        valid = bool(
            valid_alpha
            and s_info["sum"] >= 0.0
            and b_value > 0.0
            and np.isfinite(b_value)
        )

        z = asimov_z(s_info["sum"], b_value) if valid else 0.0

        if valid and z > 0.0:
            log_term = np.log1p(s_info["sum"] / b_value)
            dz_ds = log_term / z
            dz_db = (log_term - s_info["sum"] / b_value) / z
            sigma_z = float(np.hypot(
                dz_ds * s_info["err"],
                dz_db * sigma_b,
            ))
        else:
            sigma_z = 0.0

        results[name] = {
            "S": s_info["sum"],
            "sigma_S": s_info["err"],
            "N_signal": s_info["n"],
            "N_bkg_sr": y_info["n"],
            "Y_bkg_sr": y_bkg_sr,
            "sigma_Y_bkg_sr": y_info["err"],
            "neff_bkg_sr": y_info["neff"],
            "nneg_bkg_sr": y_info["nneg"],
            "alpha": float(alpha),
            "sigma_alpha": float(sigma_alpha),
            "B": float(b_value),
            "sigma_B": sigma_b,
            "Z": z,
            "sigma_Z": sigma_z,
            "valid": valid,
        }

    return results


def combined_z(results, norm, common_alpha=False):
    """
    Combine the five categories in quadrature.

    Every category's sigma_Z already includes the alpha uncertainty folded
    into sigma_B, and the categories enter the quadrature independently. This
    is the same treatment as `significance_4fold_cutbase.py`, in both alpha
    modes, so the two scripts differ only through alpha itself. (For a common
    alpha the correlation across categories is therefore not propagated
    coherently; sigma_Z is the independent-alpha approximation.)
    """
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z_total = float(np.sqrt(z2))
    if z_total <= 0.0:
        return 0.0, 0.0

    if common_alpha and not norm["common"]["valid"]:
        return z_total, float("nan")

    var_total = 0.0
    for r in results.values():
        if r["Z"] > 0.0 and np.isfinite(r["sigma_Z"]):
            var_total += (r["Z"] / z_total * r["sigma_Z"]) ** 2

    return z_total, float(np.sqrt(max(var_total, 0.0)))


# =============================================================================
# 5. Diagnostics and printing
# =============================================================================

def print_loaded_files(title, infos):
    print(f"\n{title} file coverage:")
    for process, info in infos.items():
        missing = ",".join(info["missing_eras"]) if info["missing_eras"] else "-"
        print(
            f"  {process:<34s} files={info['files']:2d} "
            f"missing_eras={missing}"
        )


def print_normalization(title, norm, common_alpha):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if common_alpha:
        c = norm["common"]
        print(
            f"Common alpha={c['alpha']:.6f} (N_Data_SB={c['N_data_sb']}, "
            f"Y_Bkg_SB={c['Y_bkg_sb']:.4f}, sigma_alpha={c['sigma_alpha']:.6f})"
        )
        return

    header = (
        f"{'Category':<12s}{'SB tag':>8s}{'N_data_SB':>12s}"
        f"{'Y_bkg_SB':>14s}{'sigma_Y_SB':>13s}"
        f"{'alpha':>11s}{'sigma_alpha':>13s}"
    )
    print(header)
    print("-" * len(header))
    for name, _ in CAT_TAGS:
        c = norm["per_cat"][name]
        print(
            f"{name:<12s}{c['sb_tag']:>8d}{c['N_data_sb']:>12d}"
            f"{c['Y_bkg_sb']:>14.4f}{c['sigma_Y_bkg_sb']:>13.4f}"
            f"{c['alpha']:>11.5f}{c['sigma_alpha']:>13.5f}"
        )


def print_table(title, results, norm, common_alpha):
    z_total, sigma_z_total = combined_z(results, norm, common_alpha)

    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    header = (
        f"{'Category':<10s}{'tag':>5s}"
        f"{'S':>11s}{'sigS':>10s}"
        f"{'N_BkgSR':>10s}{'Y_BkgSR':>12s}{'sigY_BkgSR':>13s}"
        f"{'alpha':>10s}{'B':>12s}{'sigB':>11s}"
        f"{'Z':>10s}{'sigZ':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, sr_tag in CAT_TAGS:
        r = results[name]
        print(
            f"{name:<10s}{sr_tag:>5d}"
            f"{r['S']:>11.4f}{r['sigma_S']:>10.4f}"
            f"{r['N_bkg_sr']:>10d}{r['Y_bkg_sr']:>12.4f}"
            f"{r['sigma_Y_bkg_sr']:>13.4f}"
            f"{r['alpha']:>10.5f}{r['B']:>12.4f}{r['sigma_B']:>11.4f}"
            f"{r['Z']:>10.5f}{r['sigma_Z']:>10.5f}"
        )

    print("-" * len(header))
    print(
        f"Combined Z = {z_total:.6f} ± {sigma_z_total:.6f}"
    )
    return z_total, sigma_z_total


# =============================================================================
# 6. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight-branch",
        type=str,
        default=DEFAULT_WEIGHT_BRANCH,
        choices=["weight_lumi_avg", "weight_lumi"],
        help="MC weight branch. Default: weight_lumi_avg.",
    )
    parser.add_argument(
        "--common-alpha",
        action="store_true",
        help="Use one alpha for all five categories instead of one per "
             "category. With this flag the script reproduces "
             "significance_4fold_cutbase.py exactly (same Z, same sigma).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Run a single fold (default: all four).",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Load samples and print normalisations only.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CUT-BASED 5-CATEGORY, 4-FOLD OOF SIGNIFICANCE (source_tree_tag)")
    print("=" * 100)
    print(f"Signal processes: {SIGNAL_DIRS}")
    print(f"Background processes: {BKG_DIRS}")
    print(f"MC weight branch: {args.weight_branch}")
    print(f"Categories: {CATS}")
    print(f"SR: is_SR == 1 & tag in {[tag for _, tag in CAT_TAGS]}")
    print("SB: is_SR == 0 & the matching odd tag (tag + 1)")
    print("No mass window is applied (is_SR already carries the mass selection).")
    print(
        "Background normalization: "
        + ("one common alpha per fold, from the Data/Bkg MC sideband."
           if args.common_alpha
           else "one alpha per category per fold, from that category's "
                "Data/Bkg MC sideband.")
    )
    print(
        "WARNING: verify that DYG mass-point samples and DYJetsToLL_Stitch "
        "do not overlap before interpreting the total background."
    )

    folds = [args.fold] if args.fold is not None else [0, 1, 2, 3]

    fold_results = {}
    csv_rows = []

    for fold_idx in folds:
        test_parity = TEST_PARITY[fold_idx]

        print("\n" + "=" * 90)
        print(f"FOLD {fold_idx}: test parity={test_parity}")
        print("=" * 90)

        print("\nLoading all Higgs signal processes...")
        sig_all, sig_infos = load_signal_all(fold_idx, args.weight_branch)
        print(f"  Total Signal MC events loaded: {sig_all.n}")
        print_loaded_files("Signal", sig_infos)

        print("\nLoading all seven Background MC processes...")
        bkg_all, _, bkg_infos = load_background_all(fold_idx, args.weight_branch)
        print(f"  Total Background MC events loaded: {bkg_all.n}")
        print_loaded_files("Background", bkg_infos)

        print("\nLoading nominal Data...")
        data_all, data_info = load_data_all(fold_idx)
        print(
            f"  Data events loaded: {data_all.n}; "
            f"files={data_info['files']}; "
            f"missing eras={data_info['missing_eras']}"
        )

        sig_test = select_parity(sig_all, test_parity)
        bkg_test = select_parity(bkg_all, test_parity)
        data_test = select_parity(data_all, test_parity)

        print(
            f"\nTest events: Signal={sig_test.n}, "
            f"BkgMC={bkg_test.n}, Data={data_test.n}"
        )

        norm_test = compute_normalization(data_test, bkg_test, args.common_alpha)
        print_normalization(
            f"FOLD {fold_idx} TEST", norm_test, args.common_alpha
        )

        test_results = evaluate_categories(
            sig_test, bkg_test, norm_test, args.common_alpha
        )

        z_test, sigma_z_test = print_table(
            f"FOLD {fold_idx} TEST", test_results, norm_test,
            args.common_alpha,
        )

        if args.diagnostic_only:
            continue

        fold_results[fold_idx] = {
            "norm_test": norm_test,
            "test_results": test_results,
            "Z_test": z_test,
            "sigma_Z_test": sigma_z_test,
        }

        for name, sr_tag in CAT_TAGS:
            r = test_results[name]
            cat_norm = (
                norm_test["common"] if args.common_alpha
                else norm_test["per_cat"][name]
            )
            csv_rows.append({
                "fold": fold_idx,
                "category": name,
                "sr_tag": sr_tag,
                "sb_tag": sr_tag + 1,
                "common_alpha": int(args.common_alpha),
                "N_data_sb": cat_norm["N_data_sb"],
                "N_bkg_sb": cat_norm["N_bkg_sb"],
                "Y_bkg_sb": cat_norm["Y_bkg_sb"],
                "sigma_Y_bkg_sb": cat_norm["sigma_Y_bkg_sb"],
                "alpha": cat_norm["alpha"],
                "sigma_alpha": cat_norm["sigma_alpha"],
                "S": r["S"],
                "sigma_S": r["sigma_S"],
                "N_signal": r["N_signal"],
                "N_bkg_sr": r["N_bkg_sr"],
                "Y_bkg_sr": r["Y_bkg_sr"],
                "sigma_Y_bkg_sr": r["sigma_Y_bkg_sr"],
                "B": r["B"],
                "sigma_B": r["sigma_B"],
                "Z": r["Z"],
                "sigma_Z": r["sigma_Z"],
            })

    if args.diagnostic_only:
        print("\nDiagnostic completed. No Z combination was run.")
        print(f"Output directory: {OUT_DIR}")
        return

    if not fold_results:
        raise RuntimeError("No completed folds; nothing to combine.")

    def combine(field_z, field_sigma, label):
        z = float(np.sqrt(sum(r[field_z] ** 2 for r in fold_results.values())))
        sigma = float(np.sqrt(sum(
            ((r[field_z] / z) * r[field_sigma]) ** 2
            for r in fold_results.values()
            if z > 0.0 and np.isfinite(r[field_sigma])
        ))) if z > 0.0 else float("nan")
        print(
            f"\n{label} combined Z = {z:.6f} ± {sigma:.6f}"
        )
        print(
            "  = sqrt("
            + " + ".join(
                f"{r[field_z]:.5f}^2" for r in fold_results.values()
            )
            + ")"
        )
        return z, sigma

    print("\n" + "=" * 100)
    print("PER-FOLD SUMMARY")
    print("=" * 100)

    def alpha_text(norm):
        """Common alpha, or the per-category alpha range when they differ."""
        if args.common_alpha:
            return f"{norm['common']['alpha']:.6f}"
        vals = [norm["per_cat"][name]["alpha"] for name, _ in CAT_TAGS]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return "n/a"
        return f"{min(vals):.4f}-{max(vals):.4f}"

    header = (
        f"{'Fold':<7}{'alpha_test':>20}"
        f"{'Z_test':>13}{'sigmaZ_test':>14}"
    )
    print(header)
    print("-" * len(header))
    for fold_idx, result in fold_results.items():
        print(
            f"{fold_idx:<7}"
            f"{alpha_text(result['norm_test']):>20}"
            f"{result['Z_test']:>13.6f}"
            f"{result['sigma_Z_test']:>14.6f}"
        )
    print("-" * len(header))

    z_final, sigma_final = combine(
        "Z_test", "sigma_Z_test", f"{len(fold_results)}-fold Test"
    )

    # Text summary.
    report_path = OUT_DIR / "final_z.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Cut-based (source_tree_tag) 5-category significance\n")
        report.write(f"Weight branch: {args.weight_branch}\n")
        report.write(
            "Normalization: "
            + ("one common alpha per fold, from the Data/Bkg MC sideband\n"
               if args.common_alpha
               else "one alpha per category per fold, from that "
                    "category's Data/Bkg MC sideband\n")
        )
        report.write(
            "SR: is_SR == 1; SB: is_SR == 0 (matching odd tag); "
            "no mass window\n"
        )
        report.write(
            "Categories: " + ", ".join(
                f"{name}(tag {tag})" for name, tag in CAT_TAGS
            ) + "\n\n"
        )
        report.write(header + "\n")
        for fold_idx, result in fold_results.items():
            report.write(
                f"{fold_idx:<7}"
                f"{alpha_text(result['norm_test']):>20}"
                f"{result['Z_test']:>13.6f}"
                f"{result['sigma_Z_test']:>14.6f}\n"
            )
        report.write(
            f"\n4-fold combined Test Z = {z_final:.6f} +/- {sigma_final:.6f}\n"
        )

        for fold_idx, result in fold_results.items():
            results = result["test_results"]
            norm = result["norm_test"]
            report.write(f"\nFOLD {fold_idx} TEST CATEGORY RESULTS\n")
            for name, sr_tag in CAT_TAGS:
                r = results[name]
                cat_norm = (
                    norm["common"] if args.common_alpha
                    else norm["per_cat"][name]
                )
                report.write(
                    f"{name:10s} tag={sr_tag} "
                    f"N_data_sb={cat_norm['N_data_sb']} "
                    f"alpha={cat_norm['alpha']:.6f} "
                    f"S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                    f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                    f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f}\n"
                )

    # Category-level CSV.
    csv_path = OUT_DIR / "category_yields.csv"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nReport: {report_path}")
    print(f"Category yields: {csv_path}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()