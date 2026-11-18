#!/usr/bin/env python3

"""

significance_4class_v3_bkg_4cat_independent_alpha.py

4-class, 4-fold OOF significance study for HZgamma v3 (Mll80to100),
with independent sideband normalisation for each GNN category.

Background estimate:

    alpha_i = N_Data_SB_i / Y_BkgMC_SB_i

    B_i     = alpha_i * Y_BkgMC_SR_i

All four Higgs processes (ttH, VH, ggF, VBF) are treated as signal.

The four mutually exclusive GNN categories are assigned using argmax of

[p_bkg, p_ttH, p_VH, p_ggFVBF] plus three jointly-optimised thresholds.

Each GNN category has its own alpha, calculated from Data and Bkg MC

sidebands after applying the same GNN thresholds.

Cut scan: for every fold the three thresholds are optimised on the

train+validation parities (3/4 of the sample, i.e. every parity except the

test parity). The selected thresholds are then applied to the untouched test

parity, where they only label each test event with a category; no alpha and

no Z is computed fold by fold on the test side.

Final combination: the four folds contribute four disjoint test parities, so

for each category the four test-parity samples are pooled as raw weighted

sums, one alpha_i is measured on the pooled sideband, and

    B_i = alpha_i * Y_BkgMC_SR_i,   Z_i = Asimov(S_i, B_i),

    Z_total = sqrt(sum_i Z_i^2).

Error propagation: alpha_i carries the Data Poisson term 1/N_Data_SB plus the

Bkg MC sideband term sigma_Y_SB^2/Y_SB^2 = 1/Neff_SB, both evaluated on the

pooled sums (Neff_pooled = (sum w)^2 / sum w^2 is additive because the four

test parities are disjoint). sigma_B then combines that with the pooled SR

Bkg MC statistics, and sigma_Z propagates sigma_S and sigma_B through the

Asimov formula.

SR / sideband definition:

    The upstream production already split events with source_tree_tag:

        even tag (0, 2, 4, 6, 8) -> SR,       m_llg_refit in (120, 130)

        odd  tag (1, 3, 5, 7, 9) -> sideband, m_llg_refit in [110,120) U [130,180]

    Because the tag itself enforces the mass window, no additional mass

    cut is applied here. Signal enters only through SR-tagged events, the

    Z calculation uses only Bkg MC SR-tagged events, and the sideband is

    used only to measure the scale factor alpha.

Usage:

    python significance_4class_v3_bkg_4cat_independent_alpha.py --diagnostic-only

    python significance_4class_v3_bkg_4cat_independent_alpha.py --trials 200

    python significance_4class_v3_bkg_4cat_independent_alpha.py --no-optuna \\

        --c-tth 0.3 --c-vh 0.3 --c-ggfvbf 0.3

The default MC weight branch is weight_lumi, the same as the 5-category

script. Use --weight-branch weight_lumi_avg only as a cross-check.

"""

import argparse

import csv

import sys

from pathlib import Path

import numpy as np

import uproot

try:

    import optuna

except Exception:

    optuna = None

# =============================================================================

# 1. Configuration

# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")

OUT_DIR = REPO / "tools" / "results_4class_v3_bkg_4cat_independent_alpha"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_minloss_fold"

DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_minloss_fold"

TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}

VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

TREE_NOM = "merged_SR_Sideband"

MASS = "m_llg_refit"

TAG = "source_tree_tag"

SCORE = "ttH_gnn_score"

FOLDVAR = "photon_eta_fold"

# The agreed Z-weight branch, identical to the 5-category script.

DEFAULT_WEIGHT_BRANCH = "weight_lumi"

# source_tree_tag even -> SR, odd -> sideband. The mass window is already

# applied upstream when the tag is assigned, so no mass cut is needed.

MIN_BKG_YIELD = 15.0

# Category -> score column index.

COL_BKG, COL_TTH, COL_VH, COL_GGFVBF = 0, 1, 2, 3

# 0=Other, 1=ttH-enriched, 2=VH-enriched, 3=ggF+VBF-enriched.

CATS = ["Other", "ttH", "VH", "ggFVBF"]

CAT_OTHER, CAT_TTH, CAT_VH, CAT_GGFVBF = 0, 1, 2, 3

# Every category must have a measurable alpha for a cut pair to be accepted.

# Kept identical to the 5-category script so both share one floor rule.

REQUIRED_CATS = ("Other", "ttH", "VH", "ggFVBF")

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

# Optuna: jointly optimise the three enriched-category thresholds.

CUT_MIN, CUT_MAX = 0.7, 0.99

N_TRIALS = 1000

N_STARTUP_TRIALS = 50

RANDOM_STATE = 12345

# =============================================================================

# 2. Statistics

# =============================================================================

def is_sr_tag(tag):

    """Even source_tree_tag -> signal region (m_llg_refit in 120-130 GeV)."""

    return (np.asarray(tag) % 2) == 0

def is_sb_tag(tag):

    """Odd source_tree_tag -> sideband (110-120 or 130-180 GeV)."""

    return (np.asarray(tag) % 2) == 1

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

def alpha_from_yields(n_data_sb, y_bkg_sb, sigma_y_bkg_sb):
    """alpha = N(Data SB) / Y(Bkg MC SB) and its uncertainty.

    The relative variance is the Data Poisson term plus the Bkg MC
    sideband statistics sigma_Y^2 / Y^2 = 1 / Neff_SB.  Works on any set of
    summed yields, so the same formula is used per fold and on the pooled
    test-parity sample.
    """
    if n_data_sb <= 0 or not (y_bkg_sb > 0.0) or not np.isfinite(y_bkg_sb):

        return float("nan"), float("nan"), False

    alpha = n_data_sb / y_bkg_sb

    rel_var_alpha = (
        1.0 / n_data_sb
        + (sigma_y_bkg_sb / y_bkg_sb) ** 2
    )

    sigma_alpha = alpha * np.sqrt(rel_var_alpha)

    valid = bool(np.isfinite(alpha) and alpha > 0.0)

    return float(alpha), float(sigma_alpha), valid

def bin_significance(s, sigma_s, y_bkg_sr, sigma_y_bkg_sr, alpha, sigma_alpha):
    """B, its uncertainty and the Asimov Z of one category.

    The inputs are sums and have no memory of how they were formed, so this
    function serves both a single fold and the pooled four-parity sample.

    sigma_B^2 = (Y_SR sigma_alpha)^2 + (alpha sigma_Y_SR)^2
    sigma_Z^2 = (dZ/dS sigma_S)^2 + (dZ/dB sigma_B)^2
    """
    b_value = alpha * y_bkg_sr

    sigma_b_from_alpha = y_bkg_sr * sigma_alpha
    sigma_b_from_sr_mc = alpha * sigma_y_bkg_sr
    sigma_b = float(np.hypot(sigma_b_from_alpha, sigma_b_from_sr_mc))

    valid = bool(
        np.isfinite(s)
        and np.isfinite(b_value)
        and s >= 0.0
        and b_value > 0.0
    )

    z = asimov_z(s, b_value) if valid else 0.0

    if valid and z > 0.0:

        log_term = np.log1p(s / b_value)
        dz_ds = log_term / z
        dz_db = (log_term - s / b_value) / z
        sigma_z = float(np.hypot(
            dz_ds * sigma_s,
            dz_db * sigma_b,
        ))

    else:

        sigma_z = 0.0

    return {
        "B": float(b_value),
        "sigma_B": sigma_b,
        "sigma_B_from_alpha": float(abs(sigma_b_from_alpha)),
        "sigma_B_from_SR_MC": float(abs(sigma_b_from_sr_mc)),
        "Z": float(z),
        "sigma_Z": float(sigma_z),
        "valid": valid,
    }

def compute_normalization(data_sb, bkg, c_tth, c_vh, c_ggfvbf):
    """Calculate an independent sideband scale factor for each GNN category.

    alpha_i = N(Data SB, category i) / Y(Bkg MC SB, category i).
    The Data and Bkg MC use exactly the same GNN category assignment.
    """
    if data_sb.score is None or bkg.score is None:
        raise ValueError(
            "Category-specific alpha requires GNN scores for both Data and Bkg MC."
        )

    data_cat = assign_cat(data_sb.score, c_tth, c_vh, c_ggfvbf)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh, c_ggfvbf)

    data_sb_mask = is_sb_tag(data_sb.tag)
    bkg_sb_mask = is_sb_tag(bkg.tag)

    norms = {}
    for ci, cat in enumerate(CATS):
        data_mask = data_sb_mask & (data_cat == ci)
        n_data_sb = int(np.sum(data_mask))

        bkg_mask = bkg_sb_mask & (bkg_cat == ci)
        y_bkg_sb = yield_info(bkg.weight[bkg_mask])

        norm = {
            "N_data_sb": n_data_sb,
            "N_bkg_sb": y_bkg_sb["n"],
            "Y_bkg_sb": y_bkg_sb["sum"],
            "sigma_Y_bkg_sb": y_bkg_sb["err"],
            "sumw2_bkg_sb": y_bkg_sb["sumw2"],
            "neff_bkg_sb": y_bkg_sb["neff"],
            "nneg_bkg_sb": y_bkg_sb["nneg"],
            "alpha": float("nan"),
            "sigma_alpha": float("nan"),
            "valid": False,
        }

        alpha, sigma_alpha, valid = alpha_from_yields(
            n_data_sb, y_bkg_sb["sum"], y_bkg_sb["err"]
        )

        if valid:

            norm.update({

                "alpha": alpha,

                "sigma_alpha": sigma_alpha,

                "valid": valid,

            })

        norms[cat] = norm

    return norms

def combined_z(results, norm=None):
    """Combine mutually exclusive categories with independent uncertainties.

    Each category's sigma_Z already includes its own alpha uncertainty,
    signal-MC statistical uncertainty, and SR background-MC uncertainty.
    """
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z_total = float(np.sqrt(z2))
    if z_total <= 0.0:
        return 0.0, 0.0

    var_total = sum(
        ((r["Z"] / z_total) * r["sigma_Z"]) ** 2
        for r in results.values()
        if np.isfinite(r["sigma_Z"])
    )
    return z_total, float(np.sqrt(max(var_total, 0.0)))

# =============================================================================

# 3. ROOT loading helpers

# =============================================================================

def _stack_scores_4d(sc):

    """Convert uproot std::vector<float>(4) branch to an (N,4) array."""

    sc = np.asarray(sc)

    if sc.size == 0:

        return np.empty((0, 4), dtype=np.float32)

    if sc.dtype == object or sc.ndim == 1:

        rows = [np.asarray(v, dtype=np.float32).reshape(-1) for v in sc]

        if not rows:

            return np.empty((0, 4), dtype=np.float32)

        sc = np.stack(rows, axis=0)

    sc = np.asarray(sc, dtype=np.float32)

    if sc.ndim != 2 or sc.shape[1] != 4:

        raise ValueError(

f"{SCORE} must have shape (N,4), got shape {sc.shape}"

        )

    return sc

class EventSet:

    __slots__ = ("mass", "weight", "score", "foldvar", "tag")

    def __init__(self, mass, weight, score, foldvar, tag):

        self.mass = np.asarray(mass, dtype=np.float64)

        self.weight = np.asarray(weight, dtype=np.float64)

        self.score = score

        self.foldvar = np.asarray(foldvar, dtype=np.int32)

        self.tag = np.asarray(tag, dtype=np.int32)

        n = self.mass.size

        if (self.weight.size != n or self.foldvar.size != n

                or self.tag.size != n):

            raise ValueError("EventSet arrays have inconsistent lengths")

        if self.score is not None and self.score.shape != (n, 4):

            raise ValueError(

f"Score array must have shape ({n},4), got {self.score.shape}"

            )

    @property

    def n(self):

        return self.mass.size

def empty_set(with_score=True):

    score = np.empty((0, 4), dtype=np.float32) if with_score else None

    return EventSet(

        np.array([], dtype=np.float64),

        np.array([], dtype=np.float64),

        score,

        np.array([], dtype=np.int32),

        np.array([], dtype=np.int32),

    )

def _concat_eventsets(parts, with_score=True):

    parts = [p for p in parts if p.n > 0]

    if not parts:

        return empty_set(with_score=with_score)

    scores = None

    if with_score:

        scores = np.concatenate([p.score for p in parts], axis=0)

    return EventSet(

        np.concatenate([p.mass for p in parts]),

        np.concatenate([p.weight for p in parts]),

        scores,

        np.concatenate([p.foldvar for p in parts]),

        np.concatenate([p.tag for p in parts]),

    )

def _load_one_dir(base, subdir, weight_branch, with_weight=True, with_score=True):

    """

    Load one process from one scored fold directory.

    Missing era files are reported and skipped because not every process has

    all 11 eras. An existing but unreadable file or missing required branch

    raises an error instead of silently dropping that file.

    """

    process_dir = base / subdir

    if not process_dir.is_dir():

        raise FileNotFoundError(f"Sample directory does not exist: {process_dir}")

    branches = [MASS, TAG, FOLDVAR]

    if with_score:

        branches.append(SCORE)

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

        mass = np.asarray(arrays[MASS], dtype=np.float64)

        foldvar = np.asarray(arrays[FOLDVAR], dtype=np.int32)

        tag = np.asarray(arrays[TAG], dtype=np.int32)

        if with_weight:

            weight = np.asarray(arrays[weight_branch], dtype=np.float64)

        else:

            weight = np.ones(mass.shape, dtype=np.float64)

        score = _stack_scores_4d(arrays[SCORE]) if with_score else None

        pieces.append(EventSet(mass, weight, score, foldvar, tag))

        loaded_eras.append(era)

    if not pieces:

        raise RuntimeError(

f"No ROOT files were loaded for {subdir} under {base}"

        )

    result = _concat_eventsets(pieces, with_score=with_score)

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

        eventset, info = _load_one_dir(

            base, subdir, weight_branch, with_weight=True, with_score=True

        )

        parts.append(eventset)

        infos[subdir] = info

    return _concat_eventsets(parts, with_score=True), infos

def load_background_all(fold, weight_branch):

    """Load and merge all seven requested Background MC processes."""

    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"

    parts = []

    process_sets = {}

    infos = {}

    for subdir in BKG_DIRS:

        eventset, info = _load_one_dir(

            base, subdir, weight_branch, with_weight=True, with_score=True

        )

        process_sets[subdir] = eventset

        parts.append(eventset)

        infos[subdir] = info

    return _concat_eventsets(parts, with_score=True), process_sets, infos

def load_data_sideband(fold):
    """Load nominal Data, retain SB events, and keep their GNN scores.

    Data is unweighted. Its GNN scores are required to calculate one alpha
    independently in each GNN category.
    """
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    eventset, info = _load_one_dir(
        base, "Data/Data", DEFAULT_WEIGHT_BRANCH,
        with_weight=False, with_score=True,
    )

    mask = is_sb_tag(eventset.tag)
    sideband = EventSet(
        eventset.mass[mask],
        eventset.weight[mask],
        eventset.score[mask],
        eventset.foldvar[mask],
        eventset.tag[mask],
    )
    return sideband, info

def select_parity_set(eventset, parities):

    """Keep the events whose (foldvar % 4) is one of `parities`."""

    if eventset.n == 0:

        return eventset

    mask = np.isin(eventset.foldvar % 4, np.asarray(list(parities), dtype=np.int32))

    score = eventset.score[mask] if eventset.score is not None else None

    return EventSet(

        eventset.mass[mask],

        eventset.weight[mask],

        score,

        eventset.foldvar[mask],

        eventset.tag[mask],

    )


def select_parity(eventset, parity):

    if eventset.n == 0:

        return eventset

    mask = (eventset.foldvar % 4) == parity

    score = eventset.score[mask] if eventset.score is not None else None

    return EventSet(

        eventset.mass[mask],

        eventset.weight[mask],

        score,

        eventset.foldvar[mask],

        eventset.tag[mask],

    )

# =============================================================================

# 4. Category assignment: argmax + three thresholds

# =============================================================================

def assign_cat(score4d, c_tth, c_vh, c_ggfvbf):

    """

    score4d columns: [p_bkg, p_ttH, p_VH, p_ggFVBF].

    Returns int32 category: 0=Other, 1=ttH, 2=VH, 3=ggFVBF.

    """

    if score4d is None:

        raise ValueError("Cannot assign GNN categories without scores")

    if score4d.shape[0] == 0:

        return np.array([], dtype=np.int32)

    finite_rows = np.all(np.isfinite(score4d), axis=1)

    kmax = np.zeros(score4d.shape[0], dtype=np.int32)

    kmax[finite_rows] = np.argmax(score4d[finite_rows], axis=1)

# Default: Other, including rows with invalid/non-finite scores.

    cat = np.zeros(score4d.shape[0], dtype=np.int32)

    cat[

        finite_rows

        & (kmax == COL_TTH)

        & (score4d[:, COL_TTH] > c_tth)

    ] = CAT_TTH

    cat[

        finite_rows

        & (kmax == COL_VH)

        & (score4d[:, COL_VH] > c_vh)

    ] = CAT_VH

    cat[

        finite_rows

        & (kmax == COL_GGFVBF)

        & (score4d[:, COL_GGFVBF] > c_ggfvbf)

    ] = CAT_GGFVBF

    return cat

# =============================================================================

# 5. Evaluation with category-specific Bkg-MC scale factors

# =============================================================================

def evaluate_bins(sig, bkg, norm, c_tth, c_vh, c_ggfvbf):
    """Evaluate four mutually exclusive GNN categories.

    S_i = weighted yield of all four Higgs processes in SR/category i.
    alpha_i = N(Data SB, category i) / Y(Bkg MC SB, category i).
    B_i = alpha_i * weighted yield(Bkg MC SR/category i).
    """
    sig_cat = assign_cat(sig.score, c_tth, c_vh, c_ggfvbf)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh, c_ggfvbf)

    sig_sr_mask = is_sr_tag(sig.tag)
    bkg_sr_mask = is_sr_tag(bkg.tag)
    results = {}

    for ci, cat in enumerate(CATS):
        norm_cat = norm[cat]
        if not norm_cat["valid"]:
            raise ValueError(
                f"Invalid category-specific alpha for {cat}: "
                f"N_Data_SB={norm_cat['N_data_sb']}, "
                f"Y_Bkg_SB={norm_cat['Y_bkg_sb']}"
            )

        sig_sel = sig_sr_mask & (sig_cat == ci)
        sig_info = yield_info(sig.weight[sig_sel])

        bkg_sel = bkg_sr_mask & (bkg_cat == ci)
        bkg_info = yield_info(bkg.weight[bkg_sel])

        y_bkg_sr = bkg_info["sum"]
        sigma_y_bkg_sr = bkg_info["err"]

        alpha = norm_cat["alpha"]
        sigma_alpha = norm_cat["sigma_alpha"]

        bin_stats = bin_significance(
            sig_info["sum"], sig_info["err"],
            y_bkg_sr, sigma_y_bkg_sr,
            alpha, sigma_alpha,
        )

        results[cat] = {
            "S": sig_info["sum"],
            "sigma_S": sig_info["err"],
            "N_signal": sig_info["n"],
            "N_signal_negative_weights": sig_info["nneg"],
            "N_bkg_sr": bkg_info["n"],
            "Y_bkg_sr": y_bkg_sr,
            "sigma_Y_bkg_sr": sigma_y_bkg_sr,
            "neff_bkg_sr": bkg_info["neff"],
            "nneg_bkg_sr": bkg_info["nneg"],
            "N_data_sb": norm_cat["N_data_sb"],
            "Y_bkg_sb": norm_cat["Y_bkg_sb"],
            "alpha": alpha,
            "sigma_alpha": sigma_alpha,
            "B": bin_stats["B"],
            "sigma_B": bin_stats["sigma_B"],
            "sigma_B_from_alpha": bin_stats["sigma_B_from_alpha"],
            "sigma_B_from_SR_MC": bin_stats["sigma_B_from_SR_MC"],
            "Z": bin_stats["Z"],
            "sigma_Z": bin_stats["sigma_Z"],
            "valid": bin_stats["valid"],
        }

    return results

def categories_pass_floor(results, require_all=False):
    """Reject invalid required alphas and any measurable category with B <= floor."""

    for cat, result in results.items():

        if not result["valid"]:

            if require_all or cat in REQUIRED_CATS:

                return False

            continue

        if not np.isfinite(result["B"]) or result["B"] <= MIN_BKG_YIELD:

            return False

    return True

# =============================================================================

# 5b. Pooling the four test parities: one alpha and one Z per category

# =============================================================================

TEST_SUM_KEYS = (
    "n_sig_sr", "sig_sr_sumw", "sig_sr_sumw2", "nneg_sig_sr",
    "n_bkg_sr", "bkg_sr_sumw", "bkg_sr_sumw2", "nneg_bkg_sr",
    "n_bkg_sb", "bkg_sb_sumw", "bkg_sb_sumw2", "nneg_bkg_sb",
    "n_data_sb",
)

def category_test_sums(sig, bkg, data_sb, c_tth, c_vh, c_ggfvbf):
    """Raw per-category sums of one test parity: no alpha, no Z.

    Every returned quantity is additive over disjoint event sets, so pooling
    the four folds is a plain sum.  In particular

        Neff(pool) = (sum_f sum w)^2 / (sum_f sum w^2),

    which is why sum w and sum w^2 are both stored instead of only Neff.
    Each fold contributes its own thresholds, but the four test parities
    hold disjoint events, so no double counting occurs.
    """
    sig_cat = assign_cat(sig.score, c_tth, c_vh, c_ggfvbf)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh, c_ggfvbf)
    data_cat = assign_cat(data_sb.score, c_tth, c_vh, c_ggfvbf)

    sig_sr = is_sr_tag(sig.tag)
    bkg_sr = is_sr_tag(bkg.tag)
    bkg_sb = is_sb_tag(bkg.tag)
    data_sb_mask = is_sb_tag(data_sb.tag)

    sums = {}

    for ci, cat in enumerate(CATS):

        sig_info = yield_info(sig.weight[sig_sr & (sig_cat == ci)])
        bkg_sr_info = yield_info(bkg.weight[bkg_sr & (bkg_cat == ci)])
        bkg_sb_info = yield_info(bkg.weight[bkg_sb & (bkg_cat == ci)])

        sums[cat] = {
            "n_sig_sr": sig_info["n"],
            "sig_sr_sumw": sig_info["sum"],
            "sig_sr_sumw2": sig_info["sumw2"],
            "nneg_sig_sr": sig_info["nneg"],
            "n_bkg_sr": bkg_sr_info["n"],
            "bkg_sr_sumw": bkg_sr_info["sum"],
            "bkg_sr_sumw2": bkg_sr_info["sumw2"],
            "nneg_bkg_sr": bkg_sr_info["nneg"],
            "n_bkg_sb": bkg_sb_info["n"],
            "bkg_sb_sumw": bkg_sb_info["sum"],
            "bkg_sb_sumw2": bkg_sb_info["sumw2"],
            "nneg_bkg_sb": bkg_sb_info["nneg"],
            "n_data_sb": int(np.sum(data_sb_mask & (data_cat == ci))),
        }

    return sums

def add_category_sums(accumulator, sums):

    """Add one fold's raw sums into the running pool."""

    for cat, values in sums.items():

        target = accumulator.setdefault(
            cat, {key: 0.0 for key in TEST_SUM_KEYS}
        )

        for key, value in values.items():

            target[key] += float(value)

    return accumulator

def pooled_summary(accumulator):
    """One alpha, one B and one Z per category from the pooled test parities.

        alpha_i = sum_f N(Data SB)_if / sum_f Y(Bkg MC SB)_if
        B_i     = alpha_i * sum_f Y(Bkg MC SR)_if

    The pooled sums are exact because the four test parities are disjoint.
    alpha and B then go through the same helpers used per fold, so the error
    propagation is identical: sigma_alpha from the pooled Data Poisson count
    and the pooled sideband Neff, sigma_B including the pooled SR MC error.
    """
    norms = {}
    results = {}

    for cat in CATS:

        t = accumulator[cat]

        y_bkg_sr = t["bkg_sr_sumw"]
        sigma_y_bkg_sr = float(np.sqrt(max(t["bkg_sr_sumw2"], 0.0)))
        y_bkg_sb = t["bkg_sb_sumw"]
        sigma_y_bkg_sb = float(np.sqrt(max(t["bkg_sb_sumw2"], 0.0)))
        n_data_sb = int(t["n_data_sb"])

        alpha, sigma_alpha, valid = alpha_from_yields(
            n_data_sb, y_bkg_sb, sigma_y_bkg_sb
        )

        if not valid:

            raise RuntimeError(
                f"Pooled category {cat} has no valid alpha: "
                f"N_Data_SB={n_data_sb}, Y_BkgMC_SB={y_bkg_sb:.4f}"
            )

        norms[cat] = {
            "N_data_sb": n_data_sb,
            "N_bkg_sb": int(t["n_bkg_sb"]),
            "Y_bkg_sb": y_bkg_sb,
            "sigma_Y_bkg_sb": sigma_y_bkg_sb,
            "sumw2_bkg_sb": t["bkg_sb_sumw2"],
            "neff_bkg_sb": (
                y_bkg_sb * y_bkg_sb / t["bkg_sb_sumw2"]
                if t["bkg_sb_sumw2"] > 0.0 else 0.0
            ),
            "nneg_bkg_sb": int(t["nneg_bkg_sb"]),
            "alpha": alpha,
            "sigma_alpha": sigma_alpha,
            "valid": True,
        }

        bin_stats = bin_significance(
            t["sig_sr_sumw"],
            float(np.sqrt(max(t["sig_sr_sumw2"], 0.0))),
            y_bkg_sr, sigma_y_bkg_sr,
            alpha, sigma_alpha,
        )

        results[cat] = {
            "S": t["sig_sr_sumw"],
            "sigma_S": float(np.sqrt(max(t["sig_sr_sumw2"], 0.0))),
            "N_signal": int(t["n_sig_sr"]),
            "N_signal_negative_weights": int(t["nneg_sig_sr"]),
            "N_bkg_sr": int(t["n_bkg_sr"]),
            "Y_bkg_sr": y_bkg_sr,
            "sigma_Y_bkg_sr": sigma_y_bkg_sr,
            "neff_bkg_sr": (
                y_bkg_sr * y_bkg_sr / t["bkg_sr_sumw2"]
                if t["bkg_sr_sumw2"] > 0.0 else 0.0
            ),
            "nneg_bkg_sr": int(t["nneg_bkg_sr"]),
            "N_data_sb": n_data_sb,
            "Y_bkg_sb": y_bkg_sb,
            "alpha": alpha,
            "sigma_alpha": sigma_alpha,
            "B": bin_stats["B"],
            "sigma_B": bin_stats["sigma_B"],
            "sigma_B_from_alpha": bin_stats["sigma_B_from_alpha"],
            "sigma_B_from_SR_MC": bin_stats["sigma_B_from_SR_MC"],
            "Z": bin_stats["Z"],
            "sigma_Z": bin_stats["sigma_Z"],
            "valid": bin_stats["valid"],
        }

    return norms, results

# =============================================================================

# 6. Diagnostics and printing

# =============================================================================

def print_loaded_files(title, infos):

    print(f"\n{title} file coverage:")

    for process, info in infos.items():

        missing = ",".join(info["missing_eras"]) if info["missing_eras"] else "-"

        print(

f"  {process:<34s} files={info['files']:2d} "

f"missing_eras={missing}"

        )

def print_process_diagnostics(process_sets, parity, label):

    """Print per-process SR/SB yields after selecting one eta-fold parity."""

    print(f"\nBackground process diagnostics ({label}, parity={parity})")

    header = (

f"{'Process':34s}"

f"{'N_SR':>9s}{'Y_SR':>13s}{'err_SR':>12s}{'Neff_SR':>11s}"

f"{'N_SB':>9s}{'Y_SB':>13s}{'err_SB':>12s}{'Neff_SB':>11s}"

    )

    print(header)

    print("-" * len(header))

    for process, all_events in process_sets.items():

        events = select_parity(all_events, parity)

        sr = is_sr_tag(events.tag)

        sb = is_sb_tag(events.tag)

        ysr = yield_info(events.weight[sr])

        ysb = yield_info(events.weight[sb])

        print(

f"{process:34s}"

f"{ysr['n']:9d}{ysr['sum']:13.4f}{ysr['err']:12.4f}{ysr['neff']:11.1f}"

f"{ysb['n']:9d}{ysb['sum']:13.4f}{ysb['err']:12.4f}{ysb['neff']:11.1f}"

        )

def print_normalization(title, norms):
    print(f"\n{title} — category-specific Bkg-MC normalization")
    header = (
        f"{'Category':12s}"
        f"{'N_Data_SB':>12s}"
        f"{'N_Bkg_SB':>11s}"
        f"{'Y_Bkg_SB':>13s}"
        f"{'sigY_SB':>12s}"
        f"{'Neff_SB':>10s}"
        f"{'alpha':>12s}"
        f"{'sigAlpha':>12s}"
        f"{'valid':>8s}"
    )
    print(header)
    print("-" * len(header))
    for cat in CATS:
        n = norms[cat]
        print(
            f"{cat:12s}"
            f"{n['N_data_sb']:12d}"
            f"{n['N_bkg_sb']:11d}"
            f"{n['Y_bkg_sb']:13.4f}"
            f"{n['sigma_Y_bkg_sb']:12.4f}"
            f"{n['neff_bkg_sb']:10.1f}"
            f"{n['alpha']:12.6f}"
            f"{n['sigma_alpha']:12.6f}"
            f"{str(n['valid']):>8s}"
        )
    print("-" * len(header))

def print_test_sums(title, sums):
    print(f"\n{title} — raw test-parity sums (category labels only; no alpha, no Z)")
    header = (
        f"{'Category':12s}"
        f"{'N_SigSR':>9s}{'S_SR':>11s}"
        f"{'N_BkgSR':>9s}{'Y_BkgSR':>12s}"
        f"{'N_BkgSB':>9s}{'Y_BkgSB':>12s}"
        f"{'N_DataSB':>10s}"
    )
    print(header)
    print("-" * len(header))
    for cat in CATS:
        t = sums[cat]
        print(
            f"{cat:12s}"
            f"{int(t['n_sig_sr']):9d}{t['sig_sr_sumw']:11.4f}"
            f"{int(t['n_bkg_sr']):9d}{t['bkg_sr_sumw']:12.4f}"
            f"{int(t['n_bkg_sb']):9d}{t['bkg_sb_sumw']:12.4f}"
            f"{int(t['n_data_sb']):10d}"
        )
    print("-" * len(header))

def print_bins(title, results, norm, c_tth=None, c_vh=None, c_ggfvbf=None):
    z_total, sigma_z_total = combined_z(results)

    print("\n" + "=" * 170)
    if c_tth is None:
        print(f"{title}  (one alpha per category, measured on the pooled sideband)")
    else:
        print(
            f"{title}: c_ttH={c_tth:.5f}, c_VH={c_vh:.5f}, "
            f"c_ggFVBF={c_ggfvbf:.5f}"
        )
    print("Category-specific alpha is used for each row.")
    print(
        f"Cut-scan requirements: B > {MIN_BKG_YIELD:g} in every category."
    )
    print("=" * 170)

    header = (
        f"{'Category':12s}"
        f"{'N_DataSB':>10s}{'alpha':>12s}{'sigAlpha':>12s}"
        f"{'S':>11s}{'sigS':>10s}"
        f"{'N_BkgSR':>10s}{'Y_BkgSR':>13s}{'sigY_BkgSR':>13s}{'Neff_SR':>10s}"
        f"{'B':>12s}{'sigB':>11s}"
        f"{'Z':>11s}{'sigZ':>11s}"
    )
    print(header)
    print("-" * len(header))

    for cat in CATS:
        r = results[cat]
        n = norm[cat]
        print(
            f"{cat:12s}"
            f"{n['N_data_sb']:10d}{n['alpha']:12.5f}{n['sigma_alpha']:12.5f}"
            f"{r['S']:11.4f}{r['sigma_S']:10.4f}"
            f"{r['N_bkg_sr']:10d}{r['Y_bkg_sr']:13.4f}"
            f"{r['sigma_Y_bkg_sr']:13.4f}{r['neff_bkg_sr']:10.1f}"
            f"{r['B']:12.4f}{r['sigma_B']:11.4f}"
            f"{r['Z']:11.5f}{r['sigma_Z']:11.5f}"
        )

    print("-" * len(header))
    print(
        f"Combined Z = {z_total:.6f} ± {sigma_z_total:.6f} "
        f"(four categories; independent category-alpha uncertainties)"
    )

# =============================================================================

# 7. Bayesian optimisation (train+validation scan sample -> three cuts)

# =============================================================================

def optimize_bayesian(sig_scan, bkg_scan, data_sb_scan, label=""):
    """Optimise the three thresholds on the scan sample (train+validation)."""
    if optuna is None:
        raise RuntimeError(
            "Optuna is not available; use --no-optuna with the three cut values."
        )

    def objective(trial):
        c_tth = trial.suggest_float("c_tth", CUT_MIN, CUT_MAX)
        c_vh = trial.suggest_float("c_vh", CUT_MIN, CUT_MAX)
        c_ggfvbf = trial.suggest_float("c_ggfvbf", CUT_MIN, CUT_MAX)

        # Category membership depends on the thresholds, so alpha must be
        # recalculated for every Optuna trial. A cut combination can leave a
        # category with no usable alpha (e.g. N_Data_SB <= 0 or a non-positive
        # Bkg-MC sideband yield from negative event weights), in which case
        # evaluate_bins raises ValueError. Treat that trial as unusable rather
        # than letting one bad point abort the whole scan.
        try:
            norm_trial = compute_normalization(
                data_sb_scan, bkg_scan, c_tth, c_vh, c_ggfvbf
            )
            results = evaluate_bins(
                sig_scan, bkg_scan, norm_trial, c_tth, c_vh, c_ggfvbf
            )
        except ValueError:
            return -1.0
        if not categories_pass_floor(results):
            return -1.0

        z, _ = combined_z(results)
        return z if np.isfinite(z) and z > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(
        f"\nBayesian optimisation {label}: {N_TRIALS} trials "
        "(3 cuts jointly; 4 category-specific alphas recalculated per trial)..."
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value > 0.0
    ]
    if not completed:
        raise RuntimeError(
            f"No valid Optuna trial for {label}. "
            "Check category-wise Data SB/Bkg MC SB yields or try diagnostic-only."
        )

    best = max(completed, key=lambda t: t.value)
    return (
        float(best.params["c_tth"]),
        float(best.params["c_vh"]),
        float(best.params["c_ggfvbf"]),
        study,
    )

# =============================================================================

# 8. Main

# =============================================================================

def main():
    global N_TRIALS, OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip optimisation and use the supplied three cut values.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Load samples and print category-wise SR/SB/alpha diagnostics; do not optimise.",
    )
    parser.add_argument("--c-tth", type=float, default=0.3)
    parser.add_argument("--c-vh", type=float, default=0.3)
    parser.add_argument("--c-ggfvbf", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="Output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--weight-branch",
        type=str,
        default=DEFAULT_WEIGHT_BRANCH,
        choices=["weight_lumi_avg", "weight_lumi"],
        help="MC weight branch. Default: weight_lumi (Step2 Z-weight convention).",
    )
    args = parser.parse_args()
    N_TRIALS = args.trials

    OUT_DIR = Path(args.out_dir)

    if not (0.0 <= args.c_tth <= 1.0
            and 0.0 <= args.c_vh <= 1.0
            and 0.0 <= args.c_ggfvbf <= 1.0):
        parser.error("All cut values must be between 0 and 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("4-CATEGORY SCHEME: all-Higgs signal + category-specific Bkg-MC sideband scaling")
    print("=" * 100)
    print(f"Signal processes: {SIGNAL_DIRS}")
    print(f"Background processes: {BKG_DIRS}")
    print(f"MC weight branch: {args.weight_branch}")
    print(
        f"Scan-sample requirements: B > {MIN_BKG_YIELD:g} within every category"
    )
    print(f"Categories: {CATS}")
    print("SR: source_tree_tag even (m_llg_refit already in 120-130 GeV)")
    print("SB: source_tree_tag odd  (110-120 or 130-180 GeV), used for alpha only")
    print("No additional mass-window cut is applied.")
    print("Background normalization: the cut scan uses one alpha per category on "
          "train+validation; the result uses one alpha per category measured on "
          "the pooled four test parities.")
    print("Cut scan: thresholds optimised on the 3 non-test parities, then "
          "applied to the test parity only to label events with a category.")
    print("CR Data is not used.")
    print(
        "WARNING: verify that DYG mass-point samples and DYJetsToLL_Stitch "
        "do not overlap before interpreting the total background."
    )

    fold_results = {}
    csv_rows = []
    pooled_sums = {}

    for fold_idx in range(4):
        val_parity = VALID_PARITY[fold_idx]
        test_parity = TEST_PARITY[fold_idx]

        print("\n" + "=" * 90)
        print(
            f"FOLD {fold_idx}: validation parity={val_parity}, "
            f"test parity={test_parity}"
        )
        print("=" * 90)

        print("\nLoading all Higgs signal processes...")
        sig_all, sig_infos = load_signal_all(fold_idx, args.weight_branch)
        print(f"  Total Signal MC events loaded: {sig_all.n}")
        print_loaded_files("Signal", sig_infos)

        print("\nLoading all seven Background MC processes...")
        bkg_all, bkg_process_sets, bkg_infos = load_background_all(
            fold_idx, args.weight_branch
        )
        print(f"  Total Background MC events loaded: {bkg_all.n}")
        print_loaded_files("Background", bkg_infos)

        print("\nLoading nominal Data sideband with GNN scores...")
        data_sb_all, data_info = load_data_sideband(fold_idx)
        print(
            f"  Data SB events loaded: {data_sb_all.n}; "
            f"files={data_info['files']}; "
            f"missing eras={data_info['missing_eras']}"
        )

        # Cut-scan sample: the three parities that are NOT the test parity,
        # i.e. training + validation pooled. The test parity never enters the
        # scan; it only receives the selected cuts, to label its events.
        scan_parities = [p for p in range(4) if p != test_parity]
        sig_scan = select_parity_set(sig_all, scan_parities)
        bkg_scan = select_parity_set(bkg_all, scan_parities)
        data_sb_scan = select_parity_set(data_sb_all, scan_parities)

        # Single-parity subsets are kept for the per-parity diagnostics.
        sig_val = select_parity(sig_all, val_parity)
        sig_test = select_parity(sig_all, test_parity)
        bkg_val = select_parity(bkg_all, val_parity)
        bkg_test = select_parity(bkg_all, test_parity)
        data_sb_val = select_parity(data_sb_all, val_parity)
        data_sb_test = select_parity(data_sb_all, test_parity)

        print(
            f"\nScan events (parities {scan_parities} = train+validation): "
            f"Signal={sig_scan.n}, BkgMC={bkg_scan.n}, DataSB={data_sb_scan.n}"
        )
        print(
            f"Validation-only events (parity {val_parity}): Signal={sig_val.n}, "
            f"BkgMC={bkg_val.n}, DataSB={data_sb_val.n}"
        )
        print(
            f"Test events:       Signal={sig_test.n}, "
            f"BkgMC={bkg_test.n}, DataSB={data_sb_test.n}"
        )

        print_process_diagnostics(bkg_process_sets, val_parity, "validation")
        print_process_diagnostics(bkg_process_sets, test_parity, "test")

        # Diagnostic mode uses the supplied thresholds to show category-wise
        # alpha values, without running optimisation or significance evaluation.
        if args.diagnostic_only:
            diag_norm_scan = compute_normalization(
                data_sb_scan, bkg_scan,
                args.c_tth, args.c_vh, args.c_ggfvbf,
            )
            diag_norm_test = compute_normalization(
                data_sb_test, bkg_test,
                args.c_tth, args.c_vh, args.c_ggfvbf,
            )
            print_normalization(f"FOLD {fold_idx} SCAN (train+val)", diag_norm_scan)
            print_normalization(
                f"FOLD {fold_idx} TEST PARITY (diagnostic only, not used)", diag_norm_test
            )
            print("\n[diagnostic-only] Skipping cut optimisation and Z evaluation.")
            continue

        # Select thresholds on train+validation (never on the test parity).
        if args.no_optuna:
            c_tth = args.c_tth
            c_vh = args.c_vh
            c_ggfvbf = args.c_ggfvbf
            study = None
            print(
                f"\n[no-optuna] Using fixed cuts: "
                f"ttH={c_tth}, VH={c_vh}, ggFVBF={c_ggfvbf}"
            )
        else:
            c_tth, c_vh, c_ggfvbf, study = optimize_bayesian(
                sig_scan, bkg_scan, data_sb_scan,
                label=f"fold{fold_idx} train+validation",
            )

        # Scan-sample alpha with the selected thresholds. The test parity gets
        # no alpha here: its events are only labelled with a category and
        # their raw sums are pooled over the four folds (category_test_sums).
        norm_scan = compute_normalization(
            data_sb_scan, bkg_scan, c_tth, c_vh, c_ggfvbf
        )

        invalid = [cat for cat in CATS if not norm_scan[cat]["valid"]]
        if invalid:
            raise RuntimeError(
                f"Fold {fold_idx}: invalid scan-sample alpha for categories "
                f"{invalid}. Check category-wise Data SB and Bkg MC SB yields."
            )

        print_normalization(f"FOLD {fold_idx} SCAN (train+val)", norm_scan)

        # Scan-sample result with the selected thresholds.
        scan_results = evaluate_bins(
            sig_scan, bkg_scan, norm_scan, c_tth, c_vh, c_ggfvbf
        )
        if not categories_pass_floor(scan_results):
            raise RuntimeError(
                f"Fold {fold_idx}: scan-sample thresholds do not satisfy "
                f"B > {MIN_BKG_YIELD:g} in every category."
            )

        print_bins(
            f"FOLD {fold_idx} SCAN (train+val)",
            scan_results, norm_scan, c_tth, c_vh, c_ggfvbf,
        )
        z_scan, sigma_z_scan = combined_z(scan_results)

        # Test parity: label every event with a category and keep only the raw
        # sums. No alpha and no Z is computed per fold on the test side.
        test_sums = category_test_sums(
            sig_test, bkg_test, data_sb_test, c_tth, c_vh, c_ggfvbf
        )
        add_category_sums(pooled_sums, test_sums)
        print_test_sums(f"FOLD {fold_idx} TEST (parity {test_parity})", test_sums)

        fold_results[fold_idx] = {
            "c_tth": c_tth,
            "c_vh": c_vh,
            "c_ggfvbf": c_ggfvbf,
            "norm_scan": norm_scan,
            "scan_results": scan_results,
            "Z_scan": z_scan,
            "sigma_Z_scan": sigma_z_scan,
            "test_sums": test_sums,
        }

        for cat in CATS:
            r = scan_results[cat]
            n = norm_scan[cat]
            csv_rows.append({
                "fold": fold_idx,
                "split": "scan",
                "category": cat,
                "c_tth": c_tth,
                "c_vh": c_vh,
                "c_ggfvbf": c_ggfvbf,
                "N_data_sb": n["N_data_sb"],
                "N_bkg_sb": n["N_bkg_sb"],
                "Y_bkg_sb": n["Y_bkg_sb"],
                "sigma_Y_bkg_sb": n["sigma_Y_bkg_sb"],
                "neff_bkg_sb": n["neff_bkg_sb"],
                "alpha": n["alpha"],
                "sigma_alpha": n["sigma_alpha"],
                "S": r["S"],
                "sigma_S": r["sigma_S"],
                "N_bkg_sr": r["N_bkg_sr"],
                "Y_bkg_sr": r["Y_bkg_sr"],
                "sigma_Y_bkg_sr": r["sigma_Y_bkg_sr"],
                "neff_bkg_sr": r["neff_bkg_sr"],
                "B": r["B"],
                "sigma_B": r["sigma_B"],
                "Z": r["Z"],
                "sigma_Z": r["sigma_Z"],
            })

        # Raw test-parity sums: alpha/B/Z are deliberately left empty, they
        # exist only once the four folds are pooled.
        for cat in CATS:
            t = test_sums[cat]
            csv_rows.append({
                "fold": fold_idx,
                "split": "test",
                "category": cat,
                "c_tth": c_tth,
                "c_vh": c_vh,
                "c_ggfvbf": c_ggfvbf,
                "N_data_sb": int(t["n_data_sb"]),
                "N_bkg_sb": int(t["n_bkg_sb"]),
                "Y_bkg_sb": t["bkg_sb_sumw"],
                "sigma_Y_bkg_sb": float(np.sqrt(max(t["bkg_sb_sumw2"], 0.0))),
                "neff_bkg_sb": (
                    t["bkg_sb_sumw"] ** 2 / t["bkg_sb_sumw2"]
                    if t["bkg_sb_sumw2"] > 0.0 else 0.0
                ),
                "alpha": float("nan"),
                "sigma_alpha": float("nan"),
                "S": t["sig_sr_sumw"],
                "sigma_S": float(np.sqrt(max(t["sig_sr_sumw2"], 0.0))),
                "N_bkg_sr": int(t["n_bkg_sr"]),
                "Y_bkg_sr": t["bkg_sr_sumw"],
                "sigma_Y_bkg_sr": float(np.sqrt(max(t["bkg_sr_sumw2"], 0.0))),
                "neff_bkg_sr": (
                    t["bkg_sr_sumw"] ** 2 / t["bkg_sr_sumw2"]
                    if t["bkg_sr_sumw2"] > 0.0 else 0.0
                ),
                "B": float("nan"),
                "sigma_B": float("nan"),
                "Z": float("nan"),
                "sigma_Z": float("nan"),
            })

        # Save Optuna history for reproducibility.
        if study is not None:
            study.trials_dataframe().to_csv(
                OUT_DIR / f"optuna_fold{fold_idx}.csv", index=False
            )

    if args.diagnostic_only:
        print("\nDiagnostic completed. No optimisation or final Z was run.")
        print(f"Output directory: {OUT_DIR}")
        return

    if len(fold_results) != 4:
        raise RuntimeError(
            f"Expected four completed folds, got {len(fold_results)}"
        )

    # The four test parities are disjoint, so their raw sums are added and a
    # single alpha / B / Z is computed per category on the pooled sample.
    pooled_norms, pooled_results = pooled_summary(pooled_sums)

    print_bins(
        "POOLED TEST SAMPLE (4 folds x 4 categories, disjoint events)",
        pooled_results, pooled_norms,
    )

    z_final, sigma_final = combined_z(pooled_results)

    print("\n" + "=" * 100)
    print("PER-FOLD CUTS AND TEST-PARITY CONTRIBUTIONS")
    print("=" * 100)
    header = (
        f"{'Fold':<7}{'c_ttH':>10}{'c_VH':>10}{'c_ggFVBF':>12}"
        f"{'Z_scan':>12}{'S_test':>11}{'Y_BkgSR_test':>15}{'N_DataSB_test':>15}"
    )
    print(header)
    print("-" * len(header))
    for fold_idx, result in fold_results.items():
        ts = result["test_sums"]
        print(
            f"{fold_idx:<7}"
            f"{result['c_tth']:10.5f}"
            f"{result['c_vh']:10.5f}"
            f"{result['c_ggfvbf']:12.5f}"
            f"{result['Z_scan']:12.5f}"
            f"{sum(ts[c]['sig_sr_sumw'] for c in CATS):11.4f}"
            f"{sum(ts[c]['bkg_sr_sumw'] for c in CATS):15.4f}"
            f"{int(sum(ts[c]['n_data_sb'] for c in CATS)):15d}"
        )
    print("-" * len(header))
    print("  Z_scan is the cut-scan figure of merit on train+validation, not a result.")
    print("  The last three columns are raw test-parity sums, before pooling.")

    print(f"\nPOOLED TEST Z = {z_final:.6f} ± {sigma_final:.6f}")
    print("  = sqrt(" + " + ".join(
        f"{pooled_results[cat]['Z']:.5f}^2" for cat in CATS
    ) + ")")
    print("  per category: " + ", ".join(
        f"{cat}={pooled_results[cat]['Z']:.4f}" for cat in CATS
    ))
    print("  One alpha per category, measured on the pooled test sidebands; "
          "the cuts were scanned on train+validation only.")

    # Write text summary: per-fold cuts plus the pooled per-category numbers.
    report_path = OUT_DIR / "final_z.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("4-category Bkg-MC sideband-scaled significance\n")
        report.write(f"Weight branch: {args.weight_branch}\n")
        report.write(
            "SR: source_tree_tag even (m_llg_refit already in 120-130); "
            "SB: source_tree_tag odd, used for alpha only\n\n"
        )
        report.write("Procedure\n")
        report.write(
            "  Per fold the three thresholds are scanned on train+validation "
            "(every parity except that\n"
            "  fold's test parity) and then applied to the test parity only to "
            "label events with a category.\n"
            "  No alpha and no Z is computed per fold on the test side. The "
            "four test parities are disjoint,\n"
            "  so for each category their raw sums are pooled and one alpha is "
            "measured:\n"
            "    alpha_i = sum_f N(Data SB)_if / sum_f Y(BkgMC SB)_if\n"
            "    B_i     = alpha_i * sum_f Y(BkgMC SR)_if\n"
            "    Z_i     = Asimov(S_i, B_i),  Z_total = sqrt(sum_i Z_i^2)\n\n"
        )
        report.write("Error propagation\n")
        report.write(
            "  sigma_alpha/alpha = sqrt(1/N_Data_SB + 1/Neff_SB) on the pooled "
            "sums\n"
            "  Neff_SB = (sum w)^2 / sum w^2, with sum w and sum w^2 added over "
            "the disjoint parities\n"
            "  sigma_B = sqrt((Y_SR sigma_alpha)^2 + (alpha sigma_Y_SR)^2), "
            "sigma_Y^2 = sum w^2\n"
            "  sigma_Z = sqrt((dZ/dS sigma_S)^2 + (dZ/dB sigma_B)^2)\n"
            "  sigma_Z_total = sqrt(sum_i (Z_i sigma_Zi)^2) / Z_total\n\n"
        )
        report.write(header + "\n")
        for fold_idx, result in fold_results.items():
            ts = result["test_sums"]
            report.write(
                f"{fold_idx:<7}"
                f"{result['c_tth']:10.5f}"
                f"{result['c_vh']:10.5f}"
                f"{result['c_ggfvbf']:12.5f}"
                f"{result['Z_scan']:12.5f}"
                f"{sum(ts[c]['sig_sr_sumw'] for c in CATS):11.4f}"
                f"{sum(ts[c]['bkg_sr_sumw'] for c in CATS):15.4f}"
                f"{int(sum(ts[c]['n_data_sb'] for c in CATS)):15d}\n"
            )
        report.write(
            f"\nPOOLED TEST Z = {z_final:.6f} +/- {sigma_final:.6f}\n"
        )
        report.write("  per category Z: " + ", ".join(
            f"{cat}={pooled_results[cat]['Z']:.4f}" for cat in CATS
        ) + "\n")

        report.write("\nPOOLED CATEGORY RESULTS (four test parities merged)\n")
        for cat in CATS:
            n = pooled_norms[cat]
            r = pooled_results[cat]
            report.write(
                f"{cat:10s} N_Data_SB={n['N_data_sb']:d} "
                f"Y_Bkg_SB={n['Y_bkg_sb']:.6f} +/- {n['sigma_Y_bkg_sb']:.6f} "
                f"Neff_SB={n['neff_bkg_sb']:.2f} "
                f"alpha={n['alpha']:.6f} +/- {n['sigma_alpha']:.6f} "
                f"S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                f"Y_Bkg_SR={r['Y_bkg_sr']:.6f} +/- {r['sigma_Y_bkg_sr']:.6f} "
                f"Neff_SR={r['neff_bkg_sr']:.2f} "
                f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f}\n"
            )

        report.write("\nPER-FOLD SCAN (train+val) CATEGORY RESULTS\n")
        for fold_idx, result in fold_results.items():
            report.write(f"FOLD {fold_idx}\n")
            for cat in CATS:
                n = result["norm_scan"][cat]
                r = result["scan_results"][cat]
                report.write(
                    f"{cat:10s} N_Data_SB={n['N_data_sb']:d} "
                    f"Y_Bkg_SB={n['Y_bkg_sb']:.6f} "
                    f"alpha={n['alpha']:.8f} +/- {n['sigma_alpha']:.8f} "
                    f"S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                    f"Neff_SR={r['neff_bkg_sr']:.2f} "
                    f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                    f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f}\n"
                )

    # Pooled rows: fold = -1, split = "pooled".
    for cat in CATS:
        r = pooled_results[cat]
        n = pooled_norms[cat]
        csv_rows.append({
            "fold": -1,
            "split": "pooled",
            "category": cat,
            "c_tth": float("nan"),
            "c_vh": float("nan"),
            "c_ggfvbf": float("nan"),
            "N_data_sb": n["N_data_sb"],
            "N_bkg_sb": n["N_bkg_sb"],
            "Y_bkg_sb": n["Y_bkg_sb"],
            "sigma_Y_bkg_sb": n["sigma_Y_bkg_sb"],
            "neff_bkg_sb": n["neff_bkg_sb"],
            "alpha": n["alpha"],
            "sigma_alpha": n["sigma_alpha"],
            "S": r["S"],
            "sigma_S": r["sigma_S"],
            "N_bkg_sr": r["N_bkg_sr"],
            "Y_bkg_sr": r["Y_bkg_sr"],
            "sigma_Y_bkg_sr": r["sigma_Y_bkg_sr"],
            "neff_bkg_sr": r["neff_bkg_sr"],
            "B": r["B"],
            "sigma_B": r["sigma_B"],
            "Z": r["Z"],
            "sigma_Z": r["sigma_Z"],
        })

    # Write category-level CSV.
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
