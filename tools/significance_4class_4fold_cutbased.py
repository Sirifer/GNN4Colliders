#!/usr/bin/env python3
"""
significance_5cat_4fold_cutbased_pooled_test.py

Cut-based 5-category, 4-fold OOF significance for the v3 (Mll80to100)
HZgamma production.

This script uses the same pooled-test statistical procedure as the 5-category
GNN study, but does not scan or optimise any score thresholds. Events are
assigned directly to the five original cut-based categories using
source_tree_tag:

    tag 0 -> ttHhad      tag 2 -> ttHlep      tag 4 -> untag
    tag 6 -> VH3l        tag 8 -> VHpTmiss

For each fold k, only the test parity k is selected from the corresponding
fold-k scored production. The four disjoint test parities are then merged.
There is NO per-fold alpha or Z calculation used in the final result.

Pooled background normalisation (default: one alpha per category):

    alpha_i = N_Data,SB,i(pooled four-test sample)
              / Y_BkgMC,SB,i(pooled four-test sample)

    B_i = alpha_i * Y_BkgMC,SR,i

The final significance is calculated from the pooled yields:

    Z_total = sqrt(sum_i Z_i**2)

An optional --common-alpha flag uses one common alpha across all five
categories, still measured on the pooled four-test sample. It is retained
for cross-checks; the default is category-specific alpha.

SR / SB definition (no mass window):

    is_SR == 1 -> signal region
    is_SR == 0 -> sideband
    SR category tag i uses sideband tag i+1

No additional mass window is applied, matching the source_tree_tag / is_SR
selection used by the original cut-based analysis.

Important:
- The default MC weight branch is weight_lumi.
- Data is counted, not weighted.
- Missing era files are reported and skipped, as in the original script.
- Background events listed in VETO_BKG_EVENTS are removed at load time, before
  any fold/parity selection, and reported in the log and the final report.
  The list currently holds one aMC@NLO negative-weight event that dominated
  the ttHlep SR bin. --no-veto keeps it.
- The DYG mass-point / DYJetsToLL_Stitch overlap warning is retained; the
  script does not automatically establish sample orthogonality.
- --fold runs only the selected fold's test parity and therefore produces a
  partial-test result, NOT the full four-fold result.

Usage:
    python significance_5cat_4fold_cutbased_pooled_test.py
    python significance_5cat_4fold_cutbased_pooled_test.py --common-alpha
    python significance_5cat_4fold_cutbased_pooled_test.py --fold 0
    python significance_5cat_4fold_cutbased_pooled_test.py --diagnostic-only
    python significance_5cat_4fold_cutbased_pooled_test.py --no-veto
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
OUT_DIR = REPO / "tools" / "results_5cat_4fold_cutbased_pooled_test"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"

TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}

TREE_NOM = "merged_SR_Sideband"
IS_SR = "is_SR"
FOLDVAR = "photon_eta_fold"
TREE_TAG = "source_tree_tag"

DEFAULT_WEIGHT_BRANCH = "weight_lumi"

# Cut-based categories: (category name, SR tag). SB tag = SR tag + 1.
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
    "Bkg_MC/DYJetsToLL_Stitch",
]

ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

# Event identification branches, only read for processes/eras that have a veto.
RUNVAR = "run"
LUMIVAR = "lumi"
EVENTVAR = "event"

# Event-level vetoes for background MC: (subdir, era, run, lumi, event).
#
# Each entry must match exactly one event, otherwise loading aborts.
#
# The single entry below is the largest negative-weight aMC@NLO event in the
# ttHlep SR (tag 2, is_SR 1):
#     Bkg_MC/DYJetsToLL_Stitch, 2024, weight_lumi = -1.734815,
#     genWeight = -26780.01, m_llg_refit = 124.90.
# It contributes 79% of sum(w^2) in that category: on its own it moves
# sum(w) from +2.606 (Neff 8.6) to +0.871 (Neff 0.20), and it is the source
# of the pathological ttHlep bin.  Remove it with --no-veto.
VETO_BKG_EVENTS = [
    ("Bkg_MC/DYJetsToLL_Stitch", "2024", 1, 657581, 3650887928),
]


def _veto_list(subdir):
    """Veto specs for one process, grouped by era."""
    out = {}
    for sub, era, run, lumi, event in VETO_BKG_EVENTS:
        if sub == subdir:
            out.setdefault(era, []).append(
                (int(run), int(lumi), int(event))
            )
    return out


# =============================================================================
# 2. Statistics
# =============================================================================

def asimov_z(s, b):
    """Discovery Asimov significance for one category."""
    if not np.isfinite(s) or not np.isfinite(b) or s <= 0.0 or b <= 0.0:
        return 0.0
    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def yield_info(weights):
    """Weighted yield, sumw2 uncertainty, and effective event count."""
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
    alpha = N_data / Y_MC with independent Poisson Data and finite-MC errors.

    Returns (alpha, sigma_alpha); NaN if the estimate is undefined.
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


def concat_eventsets(parts):
    """Concatenate non-empty EventSets; preserves disjoint test parities."""
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

    Missing era files are reported and skipped. Existing but unreadable files
    or missing required branches raise an error rather than being silently
    dropped.

    Background events listed in VETO_BKG_EVENTS are removed here, before any
    fold/parity selection, and reported in the returned info dict.
    """
    process_dir = base / subdir
    if not process_dir.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {process_dir}")

    branches = [IS_SR, FOLDVAR, TREE_TAG]
    if with_weight:
        branches.append(weight_branch)

    veto_specs = _veto_list(subdir)

    pieces = []
    loaded_eras = []
    missing_eras = []
    vetoed_events = []

    for era in ALL_ERAS:
        path = process_dir / f"{era}.root"
        if not path.exists():
            missing_eras.append(era)
            continue

        era_veto = veto_specs.get(era, [])
        era_branches = list(branches)
        if era_veto:
            era_branches += [RUNVAR, LUMIVAR, EVENTVAR]

        try:
            with uproot.open(str(path)) as root_file:
                if TREE_NOM not in root_file:
                    raise KeyError(f"TTree {TREE_NOM!r} not found")
                tree = root_file[TREE_NOM]
                available = set(tree.keys())
                missing_branches = [b for b in era_branches if b not in available]
                if missing_branches:
                    raise KeyError(f"Missing branches {missing_branches} in {path}")
                arrays = tree.arrays(era_branches, library="np")
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

        if era_veto:
            run_array = np.asarray(arrays[RUNVAR], dtype=np.int64)
            lumi_array = np.asarray(arrays[LUMIVAR], dtype=np.int64)
            event_array = np.asarray(arrays[EVENTVAR], dtype=np.int64)

            keep = np.ones(is_sr.shape, dtype=bool)
            for run, lumi, event in era_veto:
                hit = (
                    (run_array == run)
                    & (lumi_array == lumi)
                    & (event_array == event)
                )
                n_hit = int(np.count_nonzero(hit))
                if n_hit != 1:
                    raise RuntimeError(
                        f"Veto target (run={run}, lumi={lumi}, event={event}) "
                        f"matched {n_hit} entries in {path}; expected exactly 1."
                    )
                vetoed_events.append({
                    "process": subdir,
                    "era": era,
                    "run": run,
                    "lumi": lumi,
                    "event": event,
                    "weight": float(weight[hit][0]),
                    "tag": int(tag[hit][0]),
                    "is_sr": int(is_sr[hit][0]),
                })
                keep &= ~hit

            is_sr = is_sr[keep]
            foldvar = foldvar[keep]
            tag = tag[keep]
            weight = weight[keep]

        pieces.append(EventSet(is_sr, weight, tag, foldvar))
        loaded_eras.append(era)

    if not pieces:
        raise RuntimeError(f"No ROOT files were loaded for {subdir} under {base}")

    result = concat_eventsets(pieces)
    info = {
        "files": len(loaded_eras),
        "eras": loaded_eras,
        "missing_eras": missing_eras,
        "vetoed": vetoed_events,
    }
    return result, info


def load_signal_all(fold, weight_branch):
    """Load and merge ttH + VH + ggF + VBF Signal MC for one model fold."""
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts = []
    infos = {}
    for subdir in SIGNAL_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch)
        parts.append(eventset)
        infos[subdir] = info
    return concat_eventsets(parts), infos


def load_background_all(fold, weight_branch):
    """Load and merge all configured background MC processes for one fold."""
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts = []
    process_sets = {}
    infos = {}
    for subdir in BKG_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch)
        process_sets[subdir] = eventset
        parts.append(eventset)
        infos[subdir] = info
    return concat_eventsets(parts), process_sets, infos


def load_data_all(fold):
    """Load nominal Data (SR + SB) for one model fold; Data is unweighted."""
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    return _load_one_dir(
        base, "Data/Data", DEFAULT_WEIGHT_BRANCH, with_weight=False
    )


def select_parity(eventset, parity):
    mask = (eventset.foldvar % 4) == parity
    return EventSet(
        eventset.is_sr[mask],
        eventset.weight[mask],
        eventset.tag[mask],
        eventset.foldvar[mask],
    )


# =============================================================================
# 4. Category selections, pooled normalisation, and evaluation
# =============================================================================

def _sb_selection(eventset, sb_tag):
    return (eventset.is_sr == 0) & (eventset.tag == sb_tag)


def _sr_selection(eventset, sr_tag):
    return (eventset.is_sr == 1) & (eventset.tag == sr_tag)


def compute_normalization(data_all, bkg, common_alpha=False):
    """
    Compute alpha on the supplied sample, which in the main flow is the
    concatenation of all four test parities.

    Default: alpha_i = N_Data_SB,i / Y_BkgMC_SB,i for each category.
    With common_alpha=True: one alpha from the sum of all five SB categories.
    """
    per_cat = {}
    tot_n_data = 0
    tot_y_sb = {
        "sum": 0.0, "err": 0.0, "sumw2": 0.0,
        "n": 0, "nneg": 0, "neff": 0.0,
    }

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
        tot_y_sb["nneg"] += y_sb["nneg"]

    alpha_common, sigma_alpha_common = alpha_and_error(tot_n_data, tot_y_sb)
    tot_y_sb["neff"] = (
        tot_y_sb["sum"] ** 2 / tot_y_sb["sumw2"]
        if tot_y_sb["sumw2"] > 0.0 else 0.0
    )

    return {
        "per_cat": per_cat,
        "common": {
            "N_data_sb": tot_n_data,
            "N_bkg_sb": tot_y_sb["n"],
            "Y_bkg_sb": tot_y_sb["sum"],
            "sigma_Y_bkg_sb": tot_y_sb["err"],
            "neff_bkg_sb": tot_y_sb["neff"],
            "alpha": alpha_common,
            "sigma_alpha": sigma_alpha_common,
            "valid": bool(np.isfinite(alpha_common) and alpha_common > 0.0),
        },
    }


def evaluate_categories(sig, bkg, norm, common_alpha=False):
    """Compute pooled S_i, B_i, and Asimov Z_i for all five categories."""
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


def combined_z(results):
    """Combine mutually exclusive categories in quadrature."""
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z_total = float(np.sqrt(z2))
    if z_total <= 0.0:
        return 0.0, 0.0

    var_total = 0.0
    for r in results.values():
        if r["Z"] > 0.0 and np.isfinite(r["sigma_Z"]):
            var_total += (r["Z"] / z_total * r["sigma_Z"]) ** 2
    return z_total, float(np.sqrt(max(var_total, 0.0)))


def raw_fold_category_yields(sig, bkg, data):
    """Per-fold raw test yields for diagnostics only; no alpha and no Z."""
    rows = []
    for name, sr_tag in CAT_TAGS:
        sb_tag = sr_tag + 1
        s_sel = _sr_selection(sig, sr_tag) & np.isfinite(sig.weight)
        b_sr_sel = _sr_selection(bkg, sr_tag) & np.isfinite(bkg.weight)
        b_sb_sel = _sb_selection(bkg, sb_tag) & np.isfinite(bkg.weight)
        d_sb_sel = _sb_selection(data, sb_tag)

        s_info = yield_info(sig.weight[s_sel])
        b_sr_info = yield_info(bkg.weight[b_sr_sel])
        b_sb_info = yield_info(bkg.weight[b_sb_sel])
        rows.append({
            "category": name,
            "sr_tag": sr_tag,
            "sb_tag": sb_tag,
            "N_signal_SR": s_info["n"],
            "S_SR": s_info["sum"],
            "N_bkg_SR": b_sr_info["n"],
            "Y_bkg_SR": b_sr_info["sum"],
            "N_bkg_SB": b_sb_info["n"],
            "Y_bkg_SB": b_sb_info["sum"],
            "N_data_SB": int(np.sum(d_sb_sel)),
        })
    return rows


# =============================================================================
# 5. Diagnostics and printing
# =============================================================================

def print_loaded_files(title, infos):
    print(f"\n{title} file coverage:")
    for process, info in infos.items():
        missing = ",".join(info["missing_eras"]) if info["missing_eras"] else "-"
        print(
            f"  {process:<34s} files={info['files']:2d} "
            f"missing_eras={missing} "
            f"vetoed={len(info.get('vetoed', []))}"
        )


def print_vetoed_events(infos):
    """Report every event removed by VETO_BKG_EVENTS."""
    vetoed = []
    for info in infos.values():
        vetoed.extend(info.get("vetoed", []))
    if not vetoed:
        return

    print(f"\nEvent-level background veto: {len(vetoed)} event(s) removed")
    header = (
        f"{'process':<32s}{'era':<10s}{'tag':>5s}{'is_SR':>7s}"
        f"{'run':>10s}{'lumi':>10s}{'event':>14s}{'weight':>14s}"
    )
    print(header)
    print("-" * len(header))
    for r in vetoed:
        print(
            f"{r['process']:<32s}{r['era']:<10s}{r['tag']:>5d}{r['is_sr']:>7d}"
            f"{r['run']:>10d}{r['lumi']:>10d}{r['event']:>14d}"
            f"{r['weight']:>14.6f}"
        )
    print(f"Total weight removed: {sum(r['weight'] for r in vetoed):+.6f}")


def print_normalization(title, norm, common_alpha=False):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)

    if common_alpha:
        c = norm["common"]
        print(
            f"Common alpha={c['alpha']:.6f} +/- {c['sigma_alpha']:.6f} "
            f"(N_Data_SB={c['N_data_sb']}, Y_Bkg_SB={c['Y_bkg_sb']:.4f}, "
            f"Neff_SB={c['neff_bkg_sb']:.1f})"
        )
        return

    header = (
        f"{'Category':<12s}{'SB tag':>8s}{'N_Data_SB':>12s}"
        f"{'N_Bkg_SB':>11s}{'Y_Bkg_SB':>14s}{'err_SB':>12s}"
        f"{'Neff_SB':>11s}{'alpha':>12s}{'sigAlpha':>13s}{'rel.err':>11s}"
    )
    print(header)
    print("-" * len(header))
    for name, _ in CAT_TAGS:
        c = norm["per_cat"][name]
        rel = c["sigma_alpha"] / c["alpha"] if c["alpha"] > 0 else float("nan")
        print(
            f"{name:<12s}{c['sb_tag']:>8d}{c['N_data_sb']:>12d}"
            f"{c['N_bkg_sb']:>11d}{c['Y_bkg_sb']:>14.4f}"
            f"{c['sigma_Y_bkg_sb']:>12.4f}{c['neff_bkg_sb']:>11.1f}"
            f"{c['alpha']:>12.5f}{c['sigma_alpha']:>13.5f}{rel:>10.2%}"
        )


def print_results_table(title, results, norm, common_alpha=False):
    z_total, sigma_z_total = combined_z(results)
    print("\n" + "=" * 142)
    print(title)
    print("=" * 142)
    header = (
        f"{'Category':<11s}{'tag':>5s}{'S':>11s}{'sigS':>10s}"
        f"{'N_BkgSR':>10s}{'Y_BkgSR':>12s}{'sigY_BkgSR':>13s}"
        f"{'Neff_SR':>10s}{'alpha':>11s}{'sigAlpha':>11s}"
        f"{'B':>12s}{'sigB':>11s}{'Z':>10s}{'sigZ':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, sr_tag in CAT_TAGS:
        r = results[name]
        print(
            f"{name:<11s}{sr_tag:>5d}"
            f"{r['S']:>11.4f}{r['sigma_S']:>10.4f}"
            f"{r['N_bkg_sr']:>10d}{r['Y_bkg_sr']:>12.4f}"
            f"{r['sigma_Y_bkg_sr']:>13.4f}{r['neff_bkg_sr']:>10.1f}"
            f"{r['alpha']:>11.5f}{r['sigma_alpha']:>11.5f}"
            f"{r['B']:>12.4f}{r['sigma_B']:>11.4f}"
            f"{r['Z']:>10.5f}{r['sigma_Z']:>10.5f}"
        )
    print("-" * len(header))
    print(f"Combined Z = {z_total:.6f} +/- {sigma_z_total:.6f}")
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
        help=f"MC weight branch (default: {DEFAULT_WEIGHT_BRANCH}).",
    )
    parser.add_argument(
        "--common-alpha",
        action="store_true",
        help="Use one common alpha across all five categories, measured on "
             "the pooled selected test sample. Default: one alpha per category.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Run only one fold's test parity (partial result; default: all four).",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Load and pool test samples, print normalisations, but do not calculate Z.",
    )
    parser.add_argument(
        "--no-veto",
        action="store_true",
        help="Keep the background events listed in VETO_BKG_EVENTS instead of "
             "removing them (default: remove them).",
    )
    args = parser.parse_args()

    if args.no_veto:
        VETO_BKG_EVENTS.clear()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folds = [args.fold] if args.fold is not None else [0, 1, 2, 3]
    is_full_four_fold = len(folds) == 4

    print("=" * 108)
    print("CUT-BASED 5-CATEGORY, POOLED 4-FOLD OOF TEST SIGNIFICANCE")
    print("=" * 108)
    print(f"Signal processes: {SIGNAL_DIRS}")
    print(f"Background processes: {BKG_DIRS}")
    print(f"MC weight branch: {args.weight_branch}")
    print(f"Categories: {CATS}")
    print(f"SR: is_SR == 1 & tag in {[tag for _, tag in CAT_TAGS]}")
    print("SB: is_SR == 0 & the matching odd tag (tag + 1)")
    print("No additional mass window is applied.")
    print("No cut scan / no Optuna: events are assigned directly by source_tree_tag.")
    print(
        f"Event-level background veto: {len(VETO_BKG_EVENTS)} event(s) "
        + ("DISABLED by --no-veto." if args.no_veto else "removed before selection.")
    )
    print(
        "Alpha/Z procedure: "
        + ("one common alpha across five categories, measured after pooling test parities."
           if args.common_alpha else
           "one alpha per category, measured after pooling test parities.")
    )
    if args.fold is not None:
        print("WARNING: --fold was specified; this is a partial-test result, not the full 4-fold result.")
    print(
        "WARNING: verify that DYG mass-point samples and DYJetsToLL_Stitch "
        "do not overlap before interpreting the total background."
    )

    sig_test_parts = []
    bkg_test_parts = []
    data_test_parts = []
    raw_fold_rows = []
    coverage_by_fold = {}

    # Load each model-fold production and retain ONLY its own held-out test parity.
    for fold_idx in folds:
        test_parity = TEST_PARITY[fold_idx]
        print("\n" + "=" * 92)
        print(f"FOLD {fold_idx}: select test parity={test_parity}")
        print("=" * 92)

        print("\nLoading all Higgs signal processes...")
        sig_all, sig_infos = load_signal_all(fold_idx, args.weight_branch)
        print(f"  Total Signal MC events loaded: {sig_all.n}")
        print_loaded_files("Signal", sig_infos)

        print("\nLoading all background MC processes...")
        bkg_all, _, bkg_infos = load_background_all(fold_idx, args.weight_branch)
        print(f"  Total Background MC events loaded: {bkg_all.n}")
        print_loaded_files("Background", bkg_infos)
        if fold_idx == folds[0]:
            # The same background events sit in every fold directory, so the
            # veto report is identical for all folds; print it once.
            print_vetoed_events(bkg_infos)

        print("\nLoading nominal Data...")
        data_all, data_info = load_data_all(fold_idx)
        print(
            f"  Data events loaded: {data_all.n}; files={data_info['files']}; "
            f"missing eras={data_info['missing_eras']}"
        )

        sig_test = select_parity(sig_all, test_parity)
        bkg_test = select_parity(bkg_all, test_parity)
        data_test = select_parity(data_all, test_parity)

        print(
            f"\nTest parity {test_parity}: Signal={sig_test.n}, "
            f"BkgMC={bkg_test.n}, Data={data_test.n}"
        )

        # Store raw per-fold yields for diagnostics/comparison only.
        fold_rows = raw_fold_category_yields(sig_test, bkg_test, data_test)
        for row in fold_rows:
            row["fold"] = fold_idx
            row["test_parity"] = test_parity
            raw_fold_rows.append(row)

        sig_test_parts.append(sig_test)
        bkg_test_parts.append(bkg_test)
        data_test_parts.append(data_test)
        coverage_by_fold[fold_idx] = {
            "signal": sig_infos,
            "background": bkg_infos,
            "data": data_info,
        }

    # Pool the disjoint held-out test parities BEFORE computing alpha, B, or Z.
    sig_pooled = concat_eventsets(sig_test_parts)
    bkg_pooled = concat_eventsets(bkg_test_parts)
    data_pooled = concat_eventsets(data_test_parts)

    print("\n" + "=" * 108)
    print("POOLED TEST SAMPLE")
    print("=" * 108)
    print(f"Folds included: {folds}")
    print(f"Pooled Signal events: {sig_pooled.n}")
    print(f"Pooled Background MC events: {bkg_pooled.n}")
    print(f"Pooled Data events: {data_pooled.n}")
    print("The four test parities are pooled first; no per-fold alpha or Z is used.")

    norm_pooled = compute_normalization(
        data_pooled, bkg_pooled, common_alpha=args.common_alpha
    )
    print_normalization(
        "POOLED TEST: CATEGORY-SPECIFIC BACKGROUND NORMALISATION"
        if not args.common_alpha else
        "POOLED TEST: COMMON BACKGROUND NORMALISATION",
        norm_pooled,
        args.common_alpha,
    )

    # Save per-fold raw yields even in diagnostic-only mode.
    raw_csv_path = OUT_DIR / "per_fold_test_raw_yields.csv"
    if raw_fold_rows:
        with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_fold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_fold_rows)

    if args.diagnostic_only:
        print("\nDiagnostic completed. Alpha was calculated on the pooled test sample; Z was not calculated.")
        print(f"Per-fold raw yields: {raw_csv_path}")
        print(f"Output directory: {OUT_DIR}")
        return

    pooled_results = evaluate_categories(
        sig_pooled, bkg_pooled, norm_pooled, common_alpha=args.common_alpha
    )
    z_final, sigma_final = print_results_table(
        "FINAL POOLED TEST RESULTS (all selected test parities)",
        pooled_results,
        norm_pooled,
        args.common_alpha,
    )

    # Final category-level CSV: one row per category, with pooled alpha and Z.
    category_rows = []
    for name, sr_tag in CAT_TAGS:
        r = pooled_results[name]
        cat_norm = norm_pooled["common"] if args.common_alpha else norm_pooled["per_cat"][name]
        category_rows.append({
            "category": name,
            "sr_tag": sr_tag,
            "sb_tag": sr_tag + 1,
            "common_alpha": int(args.common_alpha),
            "N_data_sb": cat_norm["N_data_sb"],
            "N_bkg_sb": cat_norm["N_bkg_sb"],
            "Y_bkg_sb": cat_norm["Y_bkg_sb"],
            "sigma_Y_bkg_sb": cat_norm["sigma_Y_bkg_sb"],
            "neff_bkg_sb": cat_norm.get("neff_bkg_sb", float("nan")),
            "alpha": cat_norm["alpha"],
            "sigma_alpha": cat_norm["sigma_alpha"],
            "relative_alpha_error": (
                cat_norm["sigma_alpha"] / cat_norm["alpha"]
                if cat_norm["alpha"] > 0 else float("nan")
            ),
            "S": r["S"],
            "sigma_S": r["sigma_S"],
            "N_signal_SR": r["N_signal"],
            "N_bkg_sr": r["N_bkg_sr"],
            "Y_bkg_sr": r["Y_bkg_sr"],
            "sigma_Y_bkg_sr": r["sigma_Y_bkg_sr"],
            "neff_bkg_sr": r["neff_bkg_sr"],
            "B": r["B"],
            "sigma_B": r["sigma_B"],
            "Z": r["Z"],
            "sigma_Z": r["sigma_Z"],
            "valid": int(r["valid"]),
        })

    category_csv_path = OUT_DIR / "category_yields_pooled_test.csv"
    with category_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    # Human-readable report.
    report_path = OUT_DIR / "final_z_pooled_test.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Cut-based 5-category, pooled 4-fold OOF test significance\n")
        report.write(f"Folds included: {folds}\n")
        report.write(f"Weight branch: {args.weight_branch}\n")
        report.write(
            "Normalisation: "
            + ("one common alpha across all five categories, measured on pooled test\n"
               if args.common_alpha else
               "one alpha per category, measured on pooled test\n")
        )
        report.write("SR: is_SR == 1; SB: is_SR == 0 with matching odd tag; no mass window\n")
        report.write("No cut scan; categories are assigned by source_tree_tag\n")
        report.write("Categories: " + ", ".join(
            f"{name}(SR tag {tag}, SB tag {tag + 1})" for name, tag in CAT_TAGS
        ) + "\n")
        veto_rows = [
            r
            for info in coverage_by_fold[folds[0]]["background"].values()
            for r in info.get("vetoed", [])
        ]
        if veto_rows:
            report.write(
                f"Event-level background veto: {len(veto_rows)} event(s) removed "
                "before selection\n"
            )
            for r in veto_rows:
                report.write(
                    f"  {r['process']} era={r['era']} run={r['run']} "
                    f"lumi={r['lumi']} event={r['event']} tag={r['tag']} "
                    f"is_SR={r['is_sr']} weight={r['weight']:+.6f}\n"
                )
        else:
            report.write("Event-level background veto: none\n")
        report.write("\n")
        report.write(f"Pooled Signal events: {sig_pooled.n}\n")
        report.write(f"Pooled Background MC events: {bkg_pooled.n}\n")
        report.write(f"Pooled Data events: {data_pooled.n}\n\n")
        report.write(f"Combined Z = {z_final:.6f} +/- {sigma_final:.6f}\n\n")

        for name, sr_tag in CAT_TAGS:
            r = pooled_results[name]
            cat_norm = norm_pooled["common"] if args.common_alpha else norm_pooled["per_cat"][name]
            report.write(
                f"{name:10s} SRtag={sr_tag} SBtag={sr_tag + 1} "
                f"N_Data_SB={cat_norm['N_data_sb']} "
                f"Y_Bkg_SB={cat_norm['Y_bkg_sb']:.6f} "
                f"sigma_Y_Bkg_SB={cat_norm['sigma_Y_bkg_sb']:.6f} "
                f"alpha={cat_norm['alpha']:.6f} +/- {cat_norm['sigma_alpha']:.6f} "
                f"S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                f"Neff_SR={r['neff_bkg_sr']:.3f} "
                f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f} "
                f"valid={r['valid']}\n"
            )

        report.write("\nPer-fold raw test yields (no alpha, no Z):\n")
        for fold_idx in folds:
            report.write(f"\nFOLD {fold_idx}, test parity {TEST_PARITY[fold_idx]}\n")
            for row in raw_fold_rows:
                if row["fold"] == fold_idx:
                    report.write(
                        f"{row['category']:10s} "
                        f"S_SR={row['S_SR']:.6f} "
                        f"Y_Bkg_SR={row['Y_bkg_SR']:.6f} "
                        f"Y_Bkg_SB={row['Y_bkg_SB']:.6f} "
                        f"N_Data_SB={row['N_data_SB']}\n"
                    )

    print(f"\nReport: {report_path}")
    print(f"Pooled category yields: {category_csv_path}")
    print(f"Per-fold raw yields: {raw_csv_path}")
    print(f"Output directory: {OUT_DIR}")
    if not is_full_four_fold:
        print("NOTE: this result includes only a subset of folds and is not the full 4-fold result.")


if __name__ == "__main__":
    main()
