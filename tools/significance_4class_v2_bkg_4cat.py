#!/usr/bin/env python3
"""
significance_4class_v2_bkg_4cat.py

4-class, 4-fold OOF significance study for the OLD HZgamma v2 samples.

This is `significance_4class_v3_bkg_4cat.py` with the inputs swapped for the
v2 production: the scores come from the **v3-trained** 4-class checkpoints
applied to the v2 events

    MC    root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_fold{0..3}
    Data  root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_data_fold{0..3}

produced by jobs/inference/hzg_ttH_4class_on_v2_4fold_{,data_}fold{k}_inference.sbatch.
The statistics are identical to the v3 script:

    alpha = N_Data_SB / Y_BkgMC_SB
    B_i   = alpha * Y_BkgMC_SR_i

All four Higgs processes (ttH, VH, ggF, VBF) are treated as signal. The four
mutually exclusive GNN categories are assigned using argmax of
[p_bkg, p_ttH, p_VH, p_ggFVBF] plus three jointly-optimised thresholds, and
the category significances are combined in quadrature with the common-alpha
correlation propagated into sigma_Z.

SR / sideband definition:
    even source_tree_tag (0,2,4,6,8) -> SR,       m_llg_refit in (120, 130)
    odd  source_tree_tag (1,3,5,7,9) -> sideband, m_llg_refit in [110,120) U [130,180]
The tag already enforces the mass window, so no additional mass cut is made.
Signal enters only through SR-tagged events, the Z uses only Bkg MC SR-tagged
events, and the sideband only measures alpha.

What is different from v3, and why it matters
---------------------------------------------
1. Background MC is TT + DYGto2LG_10to100 only. v2 has no DY mass-point
   samples and no DYJetsToLL_Stitch, so Y_BkgMC_SB is much smaller than in v3
   and alpha comes out around 1.9 instead of around 1.7. **The Z printed here
   is therefore not comparable with tools/results_4class_v3_bkg_4cat**; it is
   only comparable with the other v2 numbers
   (results_compare_cutbase_vs_gnn is v3, results_significance_v2 is v2).
2. Six Run3 eras only: 2022preEE ... 2025. root_p2root_v2_pt_only_data also
   holds 2026.root, but it is not in the v2 MC file lists, was never scored
   (neither for the binary model nor for the 4-class one), and is left out
   here so that both models see the same events.
3. The models are applied out of domain: trained on v3, scored on v2.
4. **The shipped v2 Data has a broken `photon_eta_fold` branch** (values 0 and
   1 only, matching `int32(photon_eta*10000)` in 0 of 224 and 0 of 4346
   events), so only two of the four parities could be selected from it. The v2
   MC branch is correct. `tools/fix_v2_data_fold.py` rewrites the Data branch
   from `photon_eta` into `root_p2root_v2_pt_only_data_foldfix/`: same events,
   same sideband total (9209), but a real four-way split
   (SB per parity 2262/2270/2347/2330).

   **That fixed copy is the default Data source.** A plain

       python significance_4class_v2_bkg_4cat.py

   reads `root_p2root_v2_pt_only_data_foldfix` (same as passing
   `--data-dir <repo>/root_p2root_v2_pt_only_data_foldfix`), so every fold
   measures its own alpha (~1.93) and all four folds are optimised on their
   own validation parity; the conservative / deferred path below never
   triggers.

   To fall back to the **unfixed** scored directories (`--data-dir ""`, which
   re-enables `--data-base`/`--data-subdir`), note that parities 2 and 3 then
   have no data sideband at all. With the default --alpha-mode pool those two
   parities fall back to one alpha measured on all Data SB against the Bkg MC
   SB of the Data-covered parities only (--alpha-mode strict instead stops
   with an error, which is the v3 behaviour). The unfixed branch also
   **doubles** alpha to ~3.9, because every Data sideband event is counted in
   two parities while the background sideband is still split four ways; that
   doubles B in the signal region and makes Z too low. Numbers from that
   branch (0.383010 fixed cuts, 0.442951 conservative, 0.448067 allfolds)
   are superseded by the fold-fixed default run.

   Consequence for the optimiser in that unfixed mode: VALID_PARITY is
   {0:1, 1:2, 2:3, 3:0}, so only folds 0 (validation parity 1) and 3
   (validation parity 0) have a validation parity with its own Data. **In that
   mode only those two folds are optimised**; folds 1 and 2 inherit the mean
   of the optimised cut sets, and that inheritance is printed and recorded in
   the CSV / final_z.txt, never hidden. `--optuna-all-folds` instead lets
   every fold with a valid (possibly pooled) validation alpha optimise its own
   cuts; that is still leakage-free — alpha for a fold's validation never uses
   that parity's events, and alpha for its test parity comes from the test
   parity's own sideband — but it means two of the four cut sets are tuned
   against a normalisation borrowed from other parities.

Usage:
    # default: the fold-fixed v2 Data, all four parities, every fold optimised
    python significance_4class_v2_bkg_4cat.py
    # same, fewer trials and/or a different output directory
    python significance_4class_v2_bkg_4cat.py --trials 200 \
        --out-dir tools/results_4class_v2_bkg_4cat_trial200
    # old behaviour: the shipped (broken-fold) scored directories
    python significance_4class_v2_bkg_4cat.py --data-dir "" \
        --out-dir tools/results_4class_v2_bkg_4cat
    # ... and tune all folds there, using the pooled validation alpha
    python significance_4class_v2_bkg_4cat.py --data-dir "" --optuna-all-folds \
        --out-dir tools/results_4class_v2_bkg_4cat_allfolds
    # no optimisation at all, one fixed cut set for every fold
    python significance_4class_v2_bkg_4cat.py --no-optuna \
        --c-tth 0.705 --c-vh 0.819 --c-ggfvbf 0.221
    # just check that everything loads and alpha is sane
    python significance_4class_v2_bkg_4cat.py --diagnostic-only

The default MC weight branch is weight_lumi_avg. The v2 Data files have no
weight_lumi_avg branch (only weight, weight_lumi, weight_lumi_abs), which is
why Data is read unweighted and only counted: the sideband enters through
N(Data SB), never through a data weight.

Alternative: borrowing v3 Data to recover all four parities
-----------------------------------------------------------
The fold fix above is the right way. Before it existed, `--data-base` /
`--data-subdir` were used to point the Data side at the v3 scored
directories, which do have all four parities (they are only consulted when
`--data-dir ""` is given, since the default is the flat fold-fixed directory):

    --data-dir "" \
    --data-base <repo>/root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold \
    --data-subdir Data/Data

Only the eras in ALL_ERAS are read, so the 2016-2018 and 2026 files that the
v3 directories also contain are ignored and the year coverage stays that of
v2.

This is **not** the same Data as v2: it is a different production. Over the
six shared eras the sideband counts differ per era by up to about 15 percent
(v2 2023preBPix SB=609 vs v3 527), while the six-era total is close
(9209 vs 9373, +1.8 percent). Mixing v3 Data with v2 MC therefore combines two
selections; it is a way to get four measurable parities, not a replacement for
reproducing the v2 Data. Say so wherever the resulting Z is quoted.
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
OUT_DIR = REPO / "tools" / "results_4class_v2_bkg_4cat_foldfix"

# The fold number in the directory name is the fold whose model produced the
# scores; `scripts/inference.py` ignores Datasets[...].folding, so every file
# holds ALL events and the out-of-fold selection is the parity mask below.
MC_SCORED_BASE = REPO / "root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_fold"
DATA_SCORED_BASE = REPO / ("root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold"
                           "_data_fold")
DATA_SUBDIR = "Data"      # <DATA_SCORED_BASE><fold>/Data/<era>.root
# One flat directory of <era>.root files used for every fold (subdir ".").
# This is the DEFAULT Data source: root_p2root_v2_pt_only_data_foldfix is the
# v2 Data itself with `photon_eta_fold` recomputed (tools/fix_v2_data_fold.py),
# so all four parities exist. The Data side never enters the Z, so all four
# folds read the same files.
# Pass `--data-dir ""` to go back to the per-fold scored layout above.
DEFAULT_DATA_DIR = REPO / "root_p2root_v2_pt_only_data_foldfix"
DATA_DIR = DEFAULT_DATA_DIR

TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}
# Parities the *shipped* v2 Data covers (broken branch). The default
# root_p2root_v2_pt_only_data_foldfix copy covers all four; keep this at (0, 1)
# so the report can tell the two sources apart.
DATA_PARITIES = (0, 1)    # measured; the unfixed v2 Data has no fold % 4 >= 2

TREE_NOM = "merged_SR_Sideband"
MASS = "m_llg_refit"
TAG = "source_tree_tag"
SCORE = "ttH_gnn_score"
FOLDVAR = "photon_eta_fold"

DEFAULT_WEIGHT_BRANCH = "weight_lumi_avg"

MIN_BKG_YIELD = 10.0

# Label -> score column index.
COL_BKG, COL_TTH, COL_VH, COL_GGFVBF = 0, 1, 2, 3

# 0=Other, 1=ttH-enriched, 2=VH-enriched, 3=ggF+VBF-enriched.
CATS = ["Other", "ttH", "VH", "ggFVBF"]
CAT_OTHER, CAT_TTH, CAT_VH, CAT_GGFVBF = 0, 1, 2, 3

SIGNAL_DIRS = [
    "Signal/ttH",
    "Signal/VH",
    "Signal/ggF",
    "Signal/VBF",
]

# v2 background list: no DY mass-point samples, no DYJetsToLL_Stitch.
BKG_DIRS = [
    "Bkg_MC/TT",
    "Bkg_MC/DYGto2LG_10to100",
]

ALL_ERAS = [
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025",
]

# Optuna: jointly optimise the three enriched-category thresholds.
CUT_MIN, CUT_MAX = 0.10, 0.99
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


def compute_normalization(data_sb, bkg):
    """
    One common alpha for all four GNN categories.

    data_sb is the sideband-tagged Data for one parity (or for all parities
    when this is used as the pooled estimate); bkg is the matching Bkg MC.
    The sideband is used only here, to measure the scale factor alpha; it
    does not enter the Z calculation.
    """
    data_mask = is_sb_tag(data_sb.tag)
    n_data_sb = int(np.sum(data_mask))

    bkg_sb_mask = is_sb_tag(bkg.tag)
    y_bkg_sb = yield_info(bkg.weight[bkg_sb_mask])

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

    if n_data_sb <= 0 or y_bkg_sb["sum"] <= 0.0:
        return norm

    alpha = n_data_sb / y_bkg_sb["sum"]

    # Independent Poisson Data count and finite-MC statistical uncertainty.
    rel_var_alpha = (
        1.0 / n_data_sb
        + (y_bkg_sb["err"] / y_bkg_sb["sum"]) ** 2
    )
    sigma_alpha = alpha * np.sqrt(rel_var_alpha)

    norm.update({
        "alpha": float(alpha),
        "sigma_alpha": float(sigma_alpha),
        "valid": bool(np.isfinite(alpha) and alpha > 0.0),
    })
    return norm


def normalization_for_split(data_all, bkg_all, parity, alpha_mode="pool"):
    """
    alpha for one parity, with a fallback for parities without Data.

    The default Data source (root_p2root_v2_pt_only_data_foldfix) covers all
    four parities, so every parity measures its own alpha and the pooling
    branch never fires. The *shipped* v2 Data only covers photon_eta_fold % 4
    in {0, 1}; when you point `--data-dir ""` at those scored directories,
    parities 2 and 3 have N(Data SB) = 0 and alpha is undefined. With
    alpha_mode="pool" the replacement is one alpha measured on all Data SB
    against **the Bkg MC SB of those same parities**; with "strict" the
    invalid norm is returned and the caller stops, which is what the v3 script
    does everywhere.

    Restricting the background to the Data-covered parities matters: the Bkg
    MC has all four parities, so dividing total N(Data SB) by total
    Y(Bkg MC SB) over all four would bias alpha low by roughly a factor two.
    """
    data_split = select_parity(data_all, parity)
    bkg_split = select_parity(bkg_all, parity)

    norm = compute_normalization(data_split, bkg_split)
    norm["parity"] = parity
    norm["alpha_source"] = "per parity"
    norm["N_data_sb_parity"] = norm["N_data_sb"]
    norm["Y_bkg_sb_parity"] = norm["Y_bkg_sb"]

    if norm["valid"] or alpha_mode != "pool":
        return norm

    covered = data_parities(data_all)
    pooled = compute_normalization(data_all, select_parities(bkg_all, covered))
    pooled["parity"] = parity
    pooled["alpha_source"] = (
        "pooled over Data parities " + ",".join(str(p) for p in covered)
    )
    pooled["pooled_parities"] = covered
    # What this parity itself has, for the diagnostics printout.
    pooled["N_data_sb_parity"] = norm["N_data_sb"]
    pooled["Y_bkg_sb_parity"] = norm["Y_bkg_sb"]
    return pooled


def combined_z(results, norm):
    """
    Combine four mutually exclusive categories in quadrature.

    The uncertainty includes the common-alpha correlation:
      - independent signal MC statistical errors by category;
      - independent Bkg MC SR statistical errors by category;
      - one shared alpha uncertainty, propagated coherently.

    This is statistical-error propagation, not a profile-likelihood fit.
    """
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z_total = float(np.sqrt(z2))
    if z_total <= 0.0:
        return 0.0, 0.0

    if not norm["valid"]:
        return z_total, float("nan")

    var_total = 0.0
    alpha_derivative_sum = 0.0

    for r in results.values():
        s = r["S"]
        b = r["B"]
        if s <= 0.0 or b <= 0.0:
            continue

        log_term = np.log1p(s / b)

        # Derivatives of Z_total = sqrt(sum_i Z_i^2).
        dzt_ds = log_term / z_total
        dzt_db = (log_term - s / b) / z_total

        # Signal MC statistical contribution.
        var_total += (dzt_ds * r["sigma_S"]) ** 2

        # Bkg MC SR statistical contribution; B_i = alpha * Y_i.
        var_total += (
            dzt_db * norm["alpha"] * r["sigma_Y_bkg_sr"]
        ) ** 2

        # Common alpha contribution is added coherently after the loop.
        alpha_derivative_sum += dzt_db * r["Y_bkg_sr"]

    var_total += (alpha_derivative_sum * norm["sigma_alpha"]) ** 2
    return z_total, float(np.sqrt(max(var_total, 0.0)))


# =============================================================================
# 3. ROOT loading helpers
# =============================================================================

def mc_dir(fold):
    return REPO / f"{MC_SCORED_BASE.name}{fold}"


def data_dir(fold):
    return Path(f"{DATA_SCORED_BASE}{fold}")


def data_source(fold):
    """(base, subdir) of the Data side for one fold.

    The default DATA_DIR (the fold-fixed v2 Data) is a flat directory read by
    every fold, i.e. subdir ".". With `--data-dir ""` DATA_DIR is None and the
    per-fold scored layout <base><fold>/<DATA_SUBDIR> is used instead.
    """
    if DATA_DIR is not None:
        return Path(DATA_DIR), Path(".")
    return data_dir(fold), Path(DATA_SUBDIR)


def describe_data_source():
    if DATA_DIR is not None:
        return f"flat directory {DATA_DIR} (identical for every fold)"
    return f"{DATA_SCORED_BASE}<fold>/{DATA_SUBDIR}"


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

    Missing era files are reported and skipped, because a process does not
    have to cover every era. An existing but unreadable file, or a missing
    required branch, raises instead of silently dropping that file.
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
    """Load and merge ttH + VH + ggF + VBF Signal MC for one fold's model."""
    base = mc_dir(fold)
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
    """Load and merge the two v2 Background MC processes."""
    base = mc_dir(fold)
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


def load_data_all(fold):
    """
    Load nominal Data. It is unweighted and unscored: the sideband is only
    counted, and Data never enters the Z.
    """
    base, subdir = data_source(fold)
    eventset, info = _load_one_dir(
        base, subdir, DEFAULT_WEIGHT_BRANCH,
        with_weight=False, with_score=False,
    )
    return eventset, info


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


def select_parities(eventset, parities):
    """Keep events whose fold % 4 is in `parities` (a sorted list of ints)."""
    if eventset.n == 0:
        return eventset

    mask = np.isin(eventset.foldvar % 4, np.asarray(sorted(parities)))
    score = eventset.score[mask] if eventset.score is not None else None

    return EventSet(
        eventset.mass[mask],
        eventset.weight[mask],
        score,
        eventset.foldvar[mask],
        eventset.tag[mask],
    )


def data_parities(data_all):
    """Which fold parities the loaded Data actually covers."""
    if data_all.n == 0:
        return []
    return sorted(int(p) for p in set((data_all.foldvar % 4).tolist()))


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
# 5. Evaluation with one common Bkg-MC scale factor
# =============================================================================

def evaluate_bins(sig, bkg, norm, c_tth, c_vh, c_ggfvbf):
    """
    S_i = weighted yield of all four Higgs processes in SR/category i.
    alpha = N(Data SB) / weighted yield(Bkg MC SB), common to all categories.
    B_i = alpha * weighted yield(Bkg MC SR/category i).

    SR is selected by source_tree_tag (even); the mass window is already
    applied upstream, so no m_llg_refit cut is performed here.
    """
    if not norm["valid"]:
        empty_results = {
            cat: {
                "S": 0.0, "sigma_S": 0.0,
                "N_bkg_sr": 0, "Y_bkg_sr": 0.0,
                "sigma_Y_bkg_sr": 0.0, "neff_bkg_sr": 0.0,
                "B": 0.0, "sigma_B": 0.0,
                "Z": 0.0, "sigma_Z": 0.0, "valid": False,
            }
            for cat in CATS
        }
        return empty_results

    sig_cat = assign_cat(sig.score, c_tth, c_vh, c_ggfvbf)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh, c_ggfvbf)

    sig_sr_mask = is_sr_tag(sig.tag)
    bkg_sr_mask = is_sr_tag(bkg.tag)

    results = {}

    for ci, cat in enumerate(CATS):
        # All Higgs production modes contribute to S.
        sig_sel = sig_sr_mask & (sig_cat == ci)
        sig_info = yield_info(sig.weight[sig_sel])

        # Background MC SR yield in this GNN category.
        bkg_sel = bkg_sr_mask & (bkg_cat == ci)
        bkg_info = yield_info(bkg.weight[bkg_sel])

        y_bkg_sr = bkg_info["sum"]
        sigma_y_bkg_sr = bkg_info["err"]

        b_value = norm["alpha"] * y_bkg_sr
        sigma_b_alpha = y_bkg_sr * norm["sigma_alpha"]
        sigma_b_mc = norm["alpha"] * sigma_y_bkg_sr
        sigma_b = float(np.hypot(sigma_b_alpha, sigma_b_mc))

        valid = (
            sig_info["sum"] >= 0.0
            and b_value > 0.0
            and np.isfinite(b_value)
        )

        z = asimov_z(sig_info["sum"], b_value) if valid else 0.0

        if valid and z > 0.0:
            log_term = np.log1p(sig_info["sum"] / b_value)
            dz_ds = log_term / z
            dz_db = (log_term - sig_info["sum"] / b_value) / z
            sigma_z = float(np.hypot(
                dz_ds * sig_info["err"],
                dz_db * sigma_b,
            ))
        else:
            sigma_z = 0.0

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
            "B": float(b_value),
            "sigma_B": sigma_b,
            "sigma_B_from_alpha": float(abs(sigma_b_alpha)),
            "sigma_B_from_SR_MC": float(abs(sigma_b_mc)),
            "Z": z,
            "sigma_Z": sigma_z,
            "valid": valid,
        }

    return results


def all_categories_meet_bkg_floor(results, floor):
    """Return True only when every category has scaled Bkg yield > floor."""
    return all(
        np.isfinite(result["B"]) and result["B"] > floor
        for result in results.values()
    )


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


def print_normalization(title, norm):
    print(f"\n{title} — common Bkg-MC normalization")
    print("-" * 76)
    print(f"  parity           = {norm.get('parity', '?')}")
    print(f"  alpha source     = {norm.get('alpha_source', '?')}")
    print(f"  N(Data SB)       = {norm['N_data_sb']}")
    print(f"  N(Bkg MC SB)     = {norm['N_bkg_sb']}")
    print(f"  Y(Bkg MC SB)     = {norm['Y_bkg_sb']:.6f}")
    print(f"  sigma(Y_Bkg SB)  = {norm['sigma_Y_bkg_sb']:.6f}")
    print(f"  Neff(Bkg MC SB)  = {norm['neff_bkg_sb']:.2f}")
    print(f"  Nneg(Bkg MC SB)  = {norm['nneg_bkg_sb']}")
    print(f"  alpha            = {norm['alpha']:.8f}")
    print(f"  sigma(alpha)     = {norm['sigma_alpha']:.8f}")
    print(f"  valid            = {norm['valid']}")
    if norm.get("alpha_source", "").startswith("pooled"):
        print(
            f"  (this parity alone: N(Data SB)="
            f"{norm['N_data_sb_parity']}, "
            f"Y(Bkg MC SB)={norm['Y_bkg_sb_parity']:.4f})"
        )
    print("-" * 76)


def print_bins(title, results, norm, c_tth, c_vh, c_ggfvbf):
    z_total, sigma_z_total = combined_z(results, norm)

    print("\n" + "=" * 142)
    print(
        f"{title}: c_ttH={c_tth:.5f}, c_VH={c_vh:.5f}, "
        f"c_ggFVBF={c_ggfvbf:.5f}"
    )
    print(
        f"Common alpha={norm['alpha']:.6f} "
        f"(N_Data_SB={norm['N_data_sb']}, Y_Bkg_SB={norm['Y_bkg_sb']:.4f})"
    )
    print("=" * 142)

    header = (
        f"{'Category':12s}"
        f"{'S':>11s}{'sigS':>10s}"
        f"{'N_BkgSR':>10s}{'Y_BkgSR':>13s}{'sigY_BkgSR':>13s}"
        f"{'B':>12s}{'sigB':>11s}"
        f"{'Z':>11s}{'sigZ':>11s}"
    )
    print(header)
    print("-" * len(header))

    for cat in CATS:
        r = results[cat]
        print(
            f"{cat:12s}"
            f"{r['S']:11.4f}{r['sigma_S']:10.4f}"
            f"{r['N_bkg_sr']:10d}{r['Y_bkg_sr']:13.4f}"
            f"{r['sigma_Y_bkg_sr']:13.4f}"
            f"{r['B']:12.4f}{r['sigma_B']:11.4f}"
            f"{r['Z']:11.5f}{r['sigma_Z']:11.5f}"
        )

    print("-" * len(header))
    print(
        f"Combined Z = {z_total:.6f} ± {sigma_z_total:.6f} "
        f"(four categories; shared-alpha correlation included in sigma_Z)"
    )
    return z_total, sigma_z_total


# =============================================================================
# 7. Bayesian optimisation (validation -> three cuts)
# =============================================================================

def optimize_bayesian(sig_val, bkg_val, norm_val, label="", floor=MIN_BKG_YIELD):
    if optuna is None:
        raise RuntimeError(
            "Optuna is not available; use --no-optuna with the three cut values."
        )

    if not norm_val["valid"]:
        raise RuntimeError(
            f"{label}: invalid validation alpha. "
            "Check Data SB and weighted Bkg MC SB before optimisation."
        )

    def objective(trial):
        c_tth = trial.suggest_float("c_tth", CUT_MIN, CUT_MAX)
        c_vh = trial.suggest_float("c_vh", CUT_MIN, CUT_MAX)
        c_ggfvbf = trial.suggest_float("c_ggfvbf", CUT_MIN, CUT_MAX)

        results = evaluate_bins(
            sig_val, bkg_val, norm_val, c_tth, c_vh, c_ggfvbf
        )
        if not all_categories_meet_bkg_floor(results, floor):
            return -1.0
        z, _ = combined_z(results, norm_val)

        # Invalid alpha is handled before entering Optuna.
        # Empty/negative-background categories contribute Z=0, not infinity.
        return z if np.isfinite(z) and z > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nBayesian optimisation {label}: {N_TRIALS} trials (3 cuts jointly)...")
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
            "Check yields or try diagnostic-only first."
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
    global N_TRIALS, OUT_DIR, DATA_SCORED_BASE, DATA_SUBDIR, DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip optimisation and use the supplied three cut values "
             "for every fold.",
    )
    parser.add_argument(
        "--optuna-all-folds",
        action="store_true",
        help="Optimise every fold whose validation alpha is valid, including "
             "folds whose validation parity has no Data and therefore uses "
             "the pooled alpha. Only matters when the Data source lacks "
             "parities (--data-dir \"\" on the shipped v2 Data, where folds 1 "
             "and 2 would otherwise be deferred). With the default fold-fixed "
             "Data every fold has its own Data and is optimised anyway.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="One flat directory of <era>.root files, read identically for "
             "every fold (overrides --data-base/--data-subdir). Default: "
             "%(default)s, the v2 Data copy whose photon_eta_fold was "
             "recomputed so that all four parities exist. Pass an empty "
             "string to use the per-fold scored layout instead.",
    )
    parser.add_argument(
        "--data-base",
        type=str,
        default=str(DATA_SCORED_BASE),
        help="Prefix of the scored Data directories; the fold number is "
             "appended. Default: %(default)s. Use the v3 "
             "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold prefix "
             "together with --data-subdir Data/Data to get all four parities.",
    )
    parser.add_argument(
        "--data-subdir",
        type=str,
        default=DATA_SUBDIR,
        help="Subdirectory under each scored Data fold directory that holds "
             "<era>.root. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="Where final_z.txt, category_yields.csv and the optuna CSVs go. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Load samples and print SR/SB/alpha diagnostics; do not optimise "
             "and do not evaluate Z.",
    )
    parser.add_argument("--c-tth", type=float, default=0.3)
    parser.add_argument("--c-vh", type=float, default=0.3)
    parser.add_argument("--c-ggfvbf", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--alpha-mode",
        choices=["pool", "strict"],
        default="pool",
        help="What to do for parities without Data. Only reached when the "
             "Data source lacks parities (--data-dir \"\" on the shipped v2 "
             "Data, which covers fold %% 4 in {0,1}); the default fold-fixed "
             "Data has all four. 'pool' uses one alpha from all Data SB / all "
             "Bkg MC SB; 'strict' stops, as the v3 script does.",
    )
    parser.add_argument(
        "--min-bkg-yield",
        type=float,
        default=MIN_BKG_YIELD,
        help="Optimisation constraint: every category must have "
             "alpha*Y_BkgSR above this. Default %(default)s.",
    )
    parser.add_argument(
        "--weight-branch",
        type=str,
        default=DEFAULT_WEIGHT_BRANCH,
        choices=["weight_lumi_avg", "weight_lumi"],
        help="MC weight branch. Default: weight_lumi_avg.",
    )
    args = parser.parse_args()
    N_TRIALS = args.trials
    OUT_DIR = Path(args.out_dir)
    DATA_SCORED_BASE = args.data_base
    DATA_SUBDIR = args.data_subdir
    # Empty string means "use the per-fold scored layout" (old behaviour).
    DATA_DIR = Path(args.data_dir) if args.data_dir else None
    if DATA_DIR is not None and not DATA_DIR.is_dir():
        parser.error(
            f"--data-dir {DATA_DIR} does not exist. The default is "
            f"{DEFAULT_DATA_DIR}, built by tools/fix_v2_data_fold.py; pass "
            "--data-dir \"\" to read the per-fold scored directories instead."
        )

    if not (0.0 <= args.c_tth <= 1.0
            and 0.0 <= args.c_vh <= 1.0
            and 0.0 <= args.c_ggfvbf <= 1.0):
        parser.error("All cut values must be between 0 and 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("4-CATEGORY SCHEME: all-Higgs signal + Bkg-MC sideband scaling")
    print("                     (v2 samples, v3-trained 4-class GNN)")
    print("=" * 100)
    print(f"Signal processes: {SIGNAL_DIRS}")
    print(f"Background processes: {BKG_DIRS}")
    print(f"Eras: {ALL_ERAS}")
    print(f"MC weight branch: {args.weight_branch}")
    print(f"Categories: {CATS}")
    print("SR: source_tree_tag even (m_llg_refit already in 120-130 GeV)")
    print("SB: source_tree_tag odd  (110-120 or 130-180 GeV), scale only")
    print("No additional mass-window cut is applied.")
    print("Background normalization: one common alpha per parity/split.")
    print("CR Data is not used.")
    print(f"Data source: {describe_data_source()}")
    print(
        f"alpha mode: {args.alpha_mode}; the parity coverage of the Data files "
        "is measured per fold — see the per-fold 'fold parities present' line"
    )
    if args.no_optuna:
        optimiser_mode = "no optimisation, one fixed cut set for every fold"
    elif args.optuna_all_folds:
        optimiser_mode = (
            "optimise every fold (pooled validation alpha allowed for "
            "parities without Data)"
        )
    else:
        optimiser_mode = (
            "optimise every fold whose validation parity has its own Data "
            "(with the default fold-fixed Data that is all four); any fold "
            "without one inherits their mean cut set"
        )
    print(f"optimiser mode: {optimiser_mode}")
    print(f"output directory: {OUT_DIR}")
    print(
        "NOTE: v2 has two background samples only, so alpha is ~1.9 rather "
        "than ~1.7 and the Z is NOT comparable to the v3 results."
    )
    print(
        "NOTE: the scores come from models trained on v3, applied to v2 "
        "events; the cuts are therefore tuned out of domain as well."
    )

    fold_results = {}
    csv_rows = []
    pending = []          # folds whose validation parity has no Data
    optimised_cuts = {}   # fold -> (c_tth, c_vh, c_ggfvbf)
    parities_seen = set()  # every fold parity the Data files actually contain

    for fold_idx in range(4):
        val_parity = VALID_PARITY[fold_idx]
        test_parity = TEST_PARITY[fold_idx]

        print("\n" + "=" * 90)
        print(
            f"FOLD {fold_idx}: validation parity={val_parity}, "
            f"test parity={test_parity}"
        )
        print("=" * 90)

        print("\nLoading all four Higgs signal processes...")
        sig_all, sig_infos = load_signal_all(fold_idx, args.weight_branch)
        print(f"  Total Signal MC events loaded: {sig_all.n}")
        print_loaded_files("Signal", sig_infos)

        print("\nLoading both Background MC processes...")
        bkg_all, bkg_process_sets, bkg_infos = load_background_all(
            fold_idx, args.weight_branch
        )
        print(f"  Total Background MC events loaded: {bkg_all.n}")
        print_loaded_files("Background", bkg_infos)

        print("\nLoading nominal Data...")
        data_all, data_info = load_data_all(fold_idx)
        parities_seen.update(data_parities(data_all))
        print(
            f"  Data events loaded: {data_all.n}; "
            f"files={data_info['files']}; "
            f"missing eras={data_info['missing_eras']}; "
            f"fold parities present={sorted(set((data_all.foldvar % 4).tolist()))}"
        )

        # Split each population into the validation and test parity.
        sig_val = select_parity(sig_all, val_parity)
        sig_test = select_parity(sig_all, test_parity)
        bkg_val = select_parity(bkg_all, val_parity)
        bkg_test = select_parity(bkg_all, test_parity)

        print(
            f"\nValidation events: Signal={sig_val.n}, BkgMC={bkg_val.n}"
        )
        print(
            f"Test events:       Signal={sig_test.n}, BkgMC={bkg_test.n}"
        )

        # Print process-level yields for both parity subsets.
        print_process_diagnostics(bkg_process_sets, val_parity, "validation")
        print_process_diagnostics(bkg_process_sets, test_parity, "test")

        # Alpha is independent of the three GNN cuts, so calculate it once.
        norm_val = normalization_for_split(
            data_all, bkg_all, val_parity, args.alpha_mode
        )
        norm_test = normalization_for_split(
            data_all, bkg_all, test_parity, args.alpha_mode
        )

        print_normalization(f"FOLD {fold_idx} VALIDATION", norm_val)
        print_normalization(f"FOLD {fold_idx} TEST", norm_test)

        if args.diagnostic_only:
            print("\n[diagnostic-only] Skipping cut optimisation and Z evaluation.")
            continue

        if not norm_test["valid"]:
            raise RuntimeError(
                f"Fold {fold_idx}: invalid test alpha; stop before test evaluation."
            )

        # A fold may only be optimised on its own validation parity when that
        # parity has Data of its own. A pooled alpha is a valid normalisation,
        # but it is not this fold's measurement, so those folds are deferred
        # (and inherit the optimised cuts) unless --optuna-all-folds is given.
        val_alpha_own = norm_val.get("alpha_source") == "per parity"
        can_optimise = bool(
            norm_val["valid"] and (val_alpha_own or args.optuna_all_folds)
        )
        if not can_optimise and args.alpha_mode == "strict":
            raise RuntimeError(
                f"Fold {fold_idx}: invalid validation alpha; stop before "
                "optimisation. Use --alpha-mode pool to reuse the alpha "
                "measured on the parities that have Data."
            )

        if args.no_optuna:
            cuts = (args.c_tth, args.c_vh, args.c_ggfvbf)
            study = None
            cuts_source = "fixed (--no-optuna)"
            print(
                f"\n[no-optuna] Using fixed cuts: "
                f"ttH={cuts[0]}, VH={cuts[1]}, ggFVBF={cuts[2]}"
            )
        elif can_optimise:
            c_tth, c_vh, c_ggfvbf, study = optimize_bayesian(
                sig_val, bkg_val, norm_val,
                label=f"fold{fold_idx} validation",
                floor=args.min_bkg_yield,
            )
            cuts = (c_tth, c_vh, c_ggfvbf)
            cuts_source = "optuna on this fold's validation parity"
            optimised_cuts[fold_idx] = cuts
        else:
            print(
                f"\n[deferred] Fold {fold_idx}: validation parity "
                f"{val_parity} has no Data of its own (N(Data SB)=0), so this "
                "fold is not optimised here. It is evaluated after the folds "
                "that are, using their mean cut set. Pass "
                "--optuna-all-folds to optimise it against the pooled alpha "
                "instead."
            )
            pending.append({
                "fold": fold_idx,
                "sig_val": sig_val, "sig_test": sig_test,
                "bkg_val": bkg_val, "bkg_test": bkg_test,
                "norm_val": norm_val, "norm_test": norm_test,
            })
            continue

        fold_results[fold_idx] = evaluate_fold(
            fold_idx, cuts, cuts_source, study,
            sig_val, sig_test, bkg_val, bkg_test, norm_val, norm_test,
            args.min_bkg_yield, csv_rows,
        )

    if args.diagnostic_only:
        print("\nDiagnostic completed. No optimisation or final Z was run.")
        print(f"Output directory: {OUT_DIR}")
        return

    # ------------------------------------------------------------------
    # Folds whose validation parity has no Data inherit the mean cut set.
    # ------------------------------------------------------------------
    if pending:
        if optimised_cuts:
            inherited = tuple(
                float(np.mean([cuts[i] for cuts in optimised_cuts.values()]))
                for i in range(3)
            )
            source_folds = sorted(optimised_cuts)
            inheritance_note = (
                "inherited: mean of optuna cuts from folds "
                + ",".join(str(f) for f in source_folds)
            )
        else:
            inherited = (args.c_tth, args.c_vh, args.c_ggfvbf)
            inheritance_note = (
                "inherited: --c-tth/--c-vh/--c-ggfvbf (no fold could be "
                "optimised)"
            )

        print("\n" + "=" * 90)
        print(
            f"DEFERRED FOLDS {[p['fold'] for p in pending]}: "
            f"using c_ttH={inherited[0]:.5f}, c_VH={inherited[1]:.5f}, "
            f"c_ggFVBF={inherited[2]:.5f}"
        )
        print(f"  ({inheritance_note})")
        print("=" * 90)

        for item in pending:
            fold_idx = item["fold"]
            print(
                f"\nFOLD {fold_idx}: validation parity "
                f"{VALID_PARITY[fold_idx]} has no Data, so these cuts come "
                "from other folds and were NOT tuned here."
            )
            fold_results[fold_idx] = evaluate_fold(
                fold_idx, inherited, inheritance_note, None,
                item["sig_val"], item["sig_test"],
                item["bkg_val"], item["bkg_test"],
                item["norm_val"], item["norm_test"],
                args.min_bkg_yield, csv_rows,
            )

    if len(fold_results) != 4:
        raise RuntimeError(
            f"Expected four completed folds, got {len(fold_results)}"
        )

    fold_results = dict(sorted(fold_results.items()))

    # Four test parities are disjoint; combine their fold-level significances.
    z_final = float(np.sqrt(sum(
        r["Z_test"] ** 2 for r in fold_results.values()
    )))

    # Fold-level errors are treated as independent because test parities
    # are disjoint. Within each fold, common-alpha correlations are included.
    sigma_final = float(np.sqrt(sum(
        ((r["Z_test"] / z_final) * r["sigma_Z_test"]) ** 2
        for r in fold_results.values()
        if z_final > 0.0
        and np.isfinite(r["sigma_Z_test"])
    )))

    print("\n" + "=" * 100)
    print("PER-FOLD SUMMARY")
    print("=" * 100)
    header = (
        f"{'Fold':<7}{'c_ttH':>10}{'c_VH':>10}{'c_ggFVBF':>12}"
        f"{'alpha_val':>13}{'alpha_test':>13}"
        f"{'Z_val':>13}{'Z_test':>13}{'sigmaZ_test':>14}"
    )
    print(header)
    print("-" * len(header))

    for fold_idx, result in fold_results.items():
        print(
            f"{fold_idx:<7}"
            f"{result['c_tth']:10.5f}"
            f"{result['c_vh']:10.5f}"
            f"{result['c_ggfvbf']:12.5f}"
            f"{result['norm_val']['alpha']:13.6f}"
            f"{result['norm_test']['alpha']:13.6f}"
            f"{result['Z_val']:13.6f}"
            f"{result['Z_test']:13.6f}"
            f"{result['sigma_Z_test']:14.6f}"
        )

    print("-" * len(header))
    print(f"\n4-fold combined Test Z = {z_final:.6f} ± {sigma_final:.6f}")
    print(
        "  = sqrt("
        + " + ".join(
            f"{r['Z_test']:.5f}^2" for r in fold_results.values()
        )
        + ")"
    )

    # ------------------------------------------------------------------
    # Write text summary.
    # ------------------------------------------------------------------
    report_path = OUT_DIR / "final_z.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("4-category Bkg-MC sideband-scaled significance — v2 samples\n")
        report.write("Scores: v3-trained 4-class GNN applied to v2 events\n")
        report.write(f"Weight branch: {args.weight_branch}\n")
        report.write(f"Eras: {', '.join(ALL_ERAS)}\n")
        report.write(f"Background processes: {', '.join(BKG_DIRS)}\n")
        report.write(
            "SR: source_tree_tag even (m_llg_refit already in 120-130); "
            "SB: source_tree_tag odd, scale factor only\n"
        )
        report.write("One common alpha per fold split, shared by four categories.\n")
        report.write(f"Data source: {describe_data_source()}\n")
        report.write(
            f"Data fold parities found: {sorted(parities_seen)} "
            f"(v2 Data as shipped has {list(DATA_PARITIES)} only)\n"
        )
        report.write(f"alpha mode: {args.alpha_mode}\n")
        report.write(f"optimiser mode: {optimiser_mode}\n")
        report.write(f"optuna trials: {N_TRIALS}\n")
        report.write(f"min Bkg yield floor: {args.min_bkg_yield:g}\n")
        report.write("\n" + header + "\n")
        for fold_idx, result in fold_results.items():
            report.write(
                f"{fold_idx:<7}"
                f"{result['c_tth']:10.5f}"
                f"{result['c_vh']:10.5f}"
                f"{result['c_ggfvbf']:12.5f}"
                f"{result['norm_val']['alpha']:13.6f}"
                f"{result['norm_test']['alpha']:13.6f}"
                f"{result['Z_val']:13.6f}"
                f"{result['Z_test']:13.6f}"
                f"{result['sigma_Z_test']:14.6f}\n"
            )
            report.write(f"        cuts source: {result['cuts_source']}\n")
        report.write(
            f"\n4-fold combined Test Z = {z_final:.6f} +/- {sigma_final:.6f}\n"
        )
        report.write(
            "Not directly comparable with results_4class_v3_bkg_4cat: "
            "different background samples and a different preselection.\n"
        )
        if DATA_DIR is not None:
            report.write(
                "Data comes from one flat directory read by all four folds. "
                "With the default that directory is "
                "root_p2root_v2_pt_only_data_foldfix: the v2 Data itself with "
                "photon_eta_fold recomputed as int32(photon_eta*10000); the "
                "shipped v2 Data branch only held 0 and 1, which is why only "
                "two folds were measurable. Same events and same sideband "
                "total, but the scale factor comes from the real four-way "
                "split: the broken branch put every Data sideband event into "
                "two parities, which doubled alpha (~3.9) and therefore B in "
                "the signal region, so Z from that branch (0.383010 fixed "
                "cuts, 0.442951 conservative, 0.448067 allfolds) is too low "
                "and superseded by this run.\n"
            )
        elif set(parities_seen) - set(DATA_PARITIES):
            report.write(
                "Data was taken from a source that covers more parities "
                "than the v2 production (v3 Data on the six shared eras). "
                "That is a different preselection: per-era SB counts "
                "differ from v2 by up to ~15%, the six-era total by ~2%. "
                "alpha therefore mixes v2 MC with v3 Data; quote that "
                "whenever this Z is quoted.\n"
            )
        else:
            report.write(
                "Data came from the per-fold scored directories "
                "(--data-dir \"\"), whose photon_eta_fold holds only 0 and 1, "
                "so parities 2 and 3 had no Data sideband of their own and "
                "used a pooled alpha. These Z values are superseded by the "
                "default fold-fixed run; rerun without --data-dir \"\" to "
                "reproduce it.\n"
            )

        for fold_idx, result in fold_results.items():
            report.write(f"\nFOLD {fold_idx} TEST CATEGORY RESULTS\n")
            report.write(
                f"alpha={result['norm_test']['alpha']:.8f}, "
                f"sigma_alpha={result['norm_test']['sigma_alpha']:.8f}, "
                f"alpha source={result['norm_test'].get('alpha_source', '?')}\n"
            )
            for cat in CATS:
                r = result["test_results"][cat]
                report.write(
                    f"{cat:10s} S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                    f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                    f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f}\n"
                )

    # ------------------------------------------------------------------
    # Write category-level CSV.
    # ------------------------------------------------------------------
    csv_path = OUT_DIR / "category_yields.csv"
    csv_rows.sort(key=lambda row: (row["fold"], row["split"], CATS.index(row["category"])))
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nReport: {report_path}")
    print(f"Category yields: {csv_path}")
    print(f"Output directory: {OUT_DIR}")


def evaluate_fold(fold_idx, cuts, cuts_source, study,
                  sig_val, sig_test, bkg_val, bkg_test,
                  norm_val, norm_test, floor, csv_rows):
    """Evaluate both splits of one fold, print, and append CSV rows."""
    c_tth, c_vh, c_ggfvbf = cuts

    # Validation result with the chosen cuts. The B floor is an optimisation
    # constraint on validation; it is checked here for the inherited-cut case
    # too, because there it is not enforced by the optimiser.
    val_results = evaluate_bins(
        sig_val, bkg_val, norm_val, c_tth, c_vh, c_ggfvbf
    )
    if not all_categories_meet_bkg_floor(val_results, floor):
        below = ", ".join(
            f"{cat}(B={r['B']:.3f})"
            for cat, r in val_results.items()
            if not (np.isfinite(r["B"]) and r["B"] > floor)
        )
        print(
            f"\nWARNING: Fold {fold_idx} VALIDATION category(ies) below "
            f"B > {floor:g}: {below}. Reporting Z anyway."
        )
    z_val, sigma_z_val = print_bins(
        f"FOLD {fold_idx} VALIDATION",
        val_results, norm_val, c_tth, c_vh, c_ggfvbf,
    )

    # Independent test evaluation; the cuts are never chosen on test.
    test_results = evaluate_bins(
        sig_test, bkg_test, norm_test, c_tth, c_vh, c_ggfvbf
    )
    # Test is never used for cut selection, so a test category falling below
    # the floor is a warning, not an error.
    if not all_categories_meet_bkg_floor(test_results, floor):
        below = ", ".join(
            f"{cat}(B={r['B']:.3f})"
            for cat, r in test_results.items()
            if not (np.isfinite(r["B"]) and r["B"] > floor)
        )
        print(
            f"\nWARNING: Fold {fold_idx} TEST category(ies) below "
            f"B > {floor:g}: {below}. "
            "Reporting Z anyway (test is not used for cut selection)."
        )
    z_test, sigma_z_test = print_bins(
        f"FOLD {fold_idx} TEST",
        test_results, norm_test, c_tth, c_vh, c_ggfvbf,
    )

    for split_name, results, norm in (
        ("validation", val_results, norm_val),
        ("test", test_results, norm_test),
    ):
        for cat in CATS:
            r = results[cat]
            csv_rows.append({
                "fold": fold_idx,
                "split": split_name,
                "category": cat,
                "c_tth": c_tth,
                "c_vh": c_vh,
                "c_ggfvbf": c_ggfvbf,
                "cuts_source": cuts_source,
                "parity": norm.get("parity", ""),
                "alpha_source": norm.get("alpha_source", ""),
                "N_data_sb": norm["N_data_sb"],
                "N_data_sb_parity": norm.get("N_data_sb_parity", ""),
                "N_bkg_sb": norm["N_bkg_sb"],
                "Y_bkg_sb": norm["Y_bkg_sb"],
                "sigma_Y_bkg_sb": norm["sigma_Y_bkg_sb"],
                "alpha": norm["alpha"],
                "sigma_alpha": norm["sigma_alpha"],
                "S": r["S"],
                "sigma_S": r["sigma_S"],
                "N_bkg_sr": r["N_bkg_sr"],
                "Y_bkg_sr": r["Y_bkg_sr"],
                "sigma_Y_bkg_sr": r["sigma_Y_bkg_sr"],
                "B": r["B"],
                "sigma_B": r["sigma_B"],
                "Z": r["Z"],
                "sigma_Z": r["sigma_Z"],
            })

    # Save Optuna history for reproducibility.
    if study is not None:
        study.trials_dataframe().to_csv(
            OUT_DIR / f"optuna_fold{fold_idx}.csv", index=False
        )

    return {
        "c_tth": c_tth,
        "c_vh": c_vh,
        "c_ggfvbf": c_ggfvbf,
        "cuts_source": cuts_source,
        "norm_val": norm_val,
        "norm_test": norm_test,
        "val_results": val_results,
        "test_results": test_results,
        "Z_val": z_val,
        "sigma_Z_val": sigma_z_val,
        "Z_test": z_test,
        "sigma_Z_test": sigma_z_test,
    }


if __name__ == "__main__":
    main()
