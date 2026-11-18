#!/usr/bin/env python3
"""5-category, 4-fold OOF H->Zgamma significance study.

The five mutually exclusive categories are assigned from
[p_bkg, p_ttH, p_VH, p_ggFVBF]:
  ttH_high : argmax == ttH and p_ttH > c_ttH
  ttH_low  : argmax == ttH and p_ttH <= c_ttH
  VH_high  : argmax == VH  and p_VH  > c_VH
  VH_low   : argmax == VH  and p_VH  <= c_VH
  Other    : everything else (including argmax == bkg or ggFVBF).

The two cuts are optimised independently, one 1-D scan each:
  c_ttH maximises Z_ttH = sqrt(Z(ttH_high)^2 + Z(ttH_low)^2)
  c_VH  maximises Z_VH  = sqrt(Z(VH_high)^2  + Z(VH_low)^2)
This is valid because Other contains only bkg/ggFVBF-argmax events and is not
moved by either cut, and each cut only re-partitions its own argmax group.
Both categories in the group being scanned are required to have a measurable
alpha and B > MIN_BKG_YIELD.

For fold k, the cut scan uses the three parities other than test parity k
(train + validation pooled). The held-out test parity is never used in the
cut scan. The selected cuts are applied to that test parity only to assign
categories and accumulate raw yields. No test-parity alpha or Z is calculated
fold by fold.

After all four folds, the four disjoint test parities are pooled. One alpha
per category is measured from the pooled Data and Bkg-MC sidebands, and the
final category yields and Asimov significances are calculated from the pooled
SR yields:
  alpha_c = N(Data SB, c) / Y(BkgMC SB, c)
  B_c     = alpha_c * Y(BkgMC SR, c)
  Z_total = sqrt(sum_c Z_c**2).

The two low categories can have too few sideband Data events to measure alpha.
They are skipped (Z=0) when not measurable, unless --require-all-categories is
specified. ttH_high, VH_high and Other are required to have measurable alpha.

SR/SB are encoded upstream by source_tree_tag: even -> SR, odd -> sideband.
No additional mass cut is applied. CR Data is not used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

try:
    import optuna
except Exception:
    optuna = None


class _Tee:
    """Mirror everything written to stdout into a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except ValueError:
                pass


# =============================================================================
# 1. Configuration
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_5cat_v3_trainval_scan_pooled_test"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_minloss_fold"
DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_minloss_fold"

TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}  # diagnostics only

TREE_NOM = "merged_SR_Sideband"
MASS = "m_llg_refit"
TAG = "source_tree_tag"
SCORE = "ttH_gnn_score"
FOLDVAR = "photon_eta_fold"

# Keep the agreed Z-weight convention explicit and overridable.
DEFAULT_WEIGHT_BRANCH = "weight_lumi"

MIN_BKG_YIELD = 15

COL_BKG, COL_TTH, COL_VH, COL_GGFVBF = 0, 1, 2, 3

CATS = ["ttH_high", "ttH_low", "VH_high", "VH_low", "Other"]
CAT_TTH_HIGH, CAT_TTH_LOW = 0, 1
CAT_VH_HIGH, CAT_VH_LOW = 2, 3
CAT_OTHER = 4

# The low categories can have very sparse sidebands; these three should be
# measurable for a useful scan and final result.
REQUIRED_CATS = ("ttH_high", "VH_high", "Other")

# The two thresholds are optimised independently. Other contains only
# bkg/ggFVBF-argmax events, which neither cut can move, and c_ttH/c_VH only
# re-partition their own argmax group. The ttH and VH objectives are therefore
# separable and each is a 1-D scan.
GROUP_CATS = {
    "ttH": ("ttH_high", "ttH_low"),
    "VH": ("VH_high", "VH_low"),
}

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

CUT_MIN, CUT_MAX = 0.10, 0.99
N_TRIALS = 1000
N_STARTUP_TRIALS = 50
RANDOM_STATE = 12345


# =============================================================================
# 2. Statistics
# =============================================================================

def is_sr_tag(tag):
    return (np.asarray(tag) % 2) == 0


def is_sb_tag(tag):
    return (np.asarray(tag) % 2) == 1


def asimov_z(s, b):
    """Discovery Asimov significance for one category."""
    if not np.isfinite(s) or not np.isfinite(b) or s <= 0.0 or b <= 0.0:
        return 0.0
    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def yield_info(weights):
    """Weighted yield and MC statistical uncertainty from sum(w^2)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return {"sum": 0.0, "err": 0.0, "sumw2": 0.0,
                "n": 0, "nneg": 0, "neff": 0.0}
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
    """Return alpha and its statistical uncertainty."""
    if n_data_sb <= 0 or not np.isfinite(y_bkg_sb) or y_bkg_sb <= 0.0:
        return float("nan"), float("nan"), False
    alpha = n_data_sb / y_bkg_sb
    rel_var = 1.0 / n_data_sb + (sigma_y_bkg_sb / y_bkg_sb) ** 2
    sigma_alpha = alpha * np.sqrt(rel_var)
    valid = bool(np.isfinite(alpha) and alpha > 0.0)
    return float(alpha), float(sigma_alpha), valid


def bin_significance(s, sigma_s, y_bkg_sr, sigma_y_bkg_sr, alpha, sigma_alpha):
    """Calculate B, sigma_B, Asimov Z and propagated sigma_Z."""
    b_value = alpha * y_bkg_sr
    sigma_b_from_alpha = y_bkg_sr * sigma_alpha
    sigma_b_from_sr_mc = alpha * sigma_y_bkg_sr
    sigma_b = float(np.hypot(sigma_b_from_alpha, sigma_b_from_sr_mc))

    valid = bool(
        np.isfinite(s) and np.isfinite(b_value)
        and s >= 0.0 and b_value > 0.0
    )
    z = asimov_z(s, b_value) if valid else 0.0

    if valid and z > 0.0:
        log_term = np.log1p(s / b_value)
        dz_ds = log_term / z
        dz_db = (log_term - s / b_value) / z
        sigma_z = float(np.hypot(dz_ds * sigma_s, dz_db * sigma_b))
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


def combined_z(results):
    """Combine mutually exclusive categories in quadrature."""
    z_total = float(np.sqrt(sum(r["Z"] ** 2 for r in results.values())))
    if z_total <= 0.0:
        return 0.0, 0.0
    variance = sum(
        (r["Z"] / z_total * r["sigma_Z"]) ** 2
        for r in results.values()
        if np.isfinite(r["sigma_Z"])
    )
    return z_total, float(np.sqrt(max(variance, 0.0)))


def group_combined_z(results, group):
    """Quadrature combination restricted to one of GROUP_CATS."""
    return combined_z({cat: results[cat] for cat in GROUP_CATS[group]})


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
        raise ValueError(f"{SCORE} must have shape (N,4), got shape {sc.shape}")
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
        if self.weight.size != n or self.foldvar.size != n or self.tag.size != n:
            raise ValueError("EventSet arrays have inconsistent lengths")
        if self.score is not None and self.score.shape != (n, 4):
            raise ValueError(f"Score array must have shape ({n},4), got {self.score.shape}")

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
    scores = np.concatenate([p.score for p in parts], axis=0) if with_score else None
    return EventSet(
        np.concatenate([p.mass for p in parts]),
        np.concatenate([p.weight for p in parts]),
        scores,
        np.concatenate([p.foldvar for p in parts]),
        np.concatenate([p.tag for p in parts]),
    )


def _load_one_dir(base, subdir, weight_branch, with_weight=True, with_score=True):
    """Load one process. Missing era files are reported; broken files fail loudly."""
    process_dir = base / subdir
    if not process_dir.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {process_dir}")

    branches = [MASS, TAG, FOLDVAR]
    if with_score:
        branches.append(SCORE)
    if with_weight:
        branches.append(weight_branch)

    pieces, loaded_eras, missing_eras = [], [], []
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
                missing = [branch for branch in branches if branch not in available]
                if missing:
                    raise KeyError(f"Missing branches {missing} in {path}")
                arrays = tree.arrays(branches, library="np")
        except Exception as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        mass = np.asarray(arrays[MASS], dtype=np.float64)
        foldvar = np.asarray(arrays[FOLDVAR], dtype=np.int32)
        tag = np.asarray(arrays[TAG], dtype=np.int32)
        weight = (np.asarray(arrays[weight_branch], dtype=np.float64)
                  if with_weight else np.ones(mass.shape, dtype=np.float64))
        score = _stack_scores_4d(arrays[SCORE]) if with_score else None

        pieces.append(EventSet(mass, weight, score, foldvar, tag))
        loaded_eras.append(era)

    if not pieces:
        raise RuntimeError(f"No ROOT files were loaded for {subdir} under {base}")

    result = _concat_eventsets(pieces, with_score=with_score)
    info = {"files": len(loaded_eras), "eras": loaded_eras, "missing_eras": missing_eras}
    return result, info


def load_signal_all(fold, weight_branch):
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts, infos = [], {}
    for subdir in SIGNAL_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch,
                                       with_weight=True, with_score=True)
        parts.append(eventset)
        infos[subdir] = info
    return _concat_eventsets(parts, with_score=True), infos


def load_background_all(fold, weight_branch):
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    parts, process_sets, infos = [], {}, {}
    for subdir in BKG_DIRS:
        eventset, info = _load_one_dir(base, subdir, weight_branch,
                                       with_weight=True, with_score=True)
        process_sets[subdir] = eventset
        parts.append(eventset)
        infos[subdir] = info
    return _concat_eventsets(parts, with_score=True), process_sets, infos


def load_data_sideband(fold):
    """Load nominal Data and retain sideband-tagged events; Data is unweighted."""
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    eventset, info = _load_one_dir(base, "Data/Data", DEFAULT_WEIGHT_BRANCH,
                                   with_weight=False, with_score=True)
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
    """Select events whose photon_eta_fold % 4 is in the supplied parity set."""
    if eventset.n == 0:
        return eventset
    mask = np.isin(eventset.foldvar % 4, np.asarray(list(parities), dtype=np.int32))
    score = eventset.score[mask] if eventset.score is not None else None
    return EventSet(eventset.mass[mask], eventset.weight[mask], score,
                    eventset.foldvar[mask], eventset.tag[mask])


def select_parity(eventset, parity):
    return select_parity_set(eventset, [parity])


# =============================================================================
# 4. Five-category assignment
# =============================================================================

def assign_cat(score4d, c_tth, c_vh):
    """Assign ids: 0 ttH_high, 1 ttH_low, 2 VH_high, 3 VH_low, 4 Other."""
    if score4d is None:
        raise ValueError("Cannot assign GNN categories without scores")
    if score4d.shape[0] == 0:
        return np.array([], dtype=np.int32)

    finite_rows = np.all(np.isfinite(score4d), axis=1)
    kmax = np.zeros(score4d.shape[0], dtype=np.int32)
    kmax[finite_rows] = np.argmax(score4d[finite_rows], axis=1)

    # Default: Other, including non-finite score rows and ggFVBF argmax.
    cat = np.full(score4d.shape[0], CAT_OTHER, dtype=np.int32)
    tth_argmax = finite_rows & (kmax == COL_TTH)
    vh_argmax = finite_rows & (kmax == COL_VH)

    cat[tth_argmax & (score4d[:, COL_TTH] > c_tth)] = CAT_TTH_HIGH
    cat[tth_argmax & (score4d[:, COL_TTH] <= c_tth)] = CAT_TTH_LOW
    cat[vh_argmax & (score4d[:, COL_VH] > c_vh)] = CAT_VH_HIGH
    cat[vh_argmax & (score4d[:, COL_VH] <= c_vh)] = CAT_VH_LOW
    return cat


# =============================================================================
# 5. Per-category sideband normalisation and SR evaluation
# =============================================================================

def empty_norm(label):
    return {
        "category": label,
        "N_data_sb": 0,
        "N_bkg_sb": 0,
        "Y_bkg_sb": 0.0,
        "sigma_Y_bkg_sb": 0.0,
        "sumw2_bkg_sb": 0.0,
        "neff_bkg_sb": 0.0,
        "nneg_bkg_sb": 0,
        "alpha": float("nan"),
        "sigma_alpha": float("nan"),
        "valid": False,
        "reason": "not_evaluated",
    }


def compute_normalization(data_sb, bkg, c_tth, c_vh):
    """Measure one alpha per category from the supplied Data/Bkg-MC sidebands."""
    data_cat = assign_cat(data_sb.score, c_tth, c_vh)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh)
    data_sb_mask = is_sb_tag(data_sb.tag)
    bkg_sb_mask = is_sb_tag(bkg.tag)

    norms = {}
    for ci, cat in enumerate(CATS):
        norm = empty_norm(cat)
        n_data_sb = int(np.sum(data_sb_mask & (data_cat == ci)))
        bkg_info = yield_info(bkg.weight[bkg_sb_mask & (bkg_cat == ci)])
        norm.update({
            "N_data_sb": n_data_sb,
            "N_bkg_sb": bkg_info["n"],
            "Y_bkg_sb": bkg_info["sum"],
            "sigma_Y_bkg_sb": bkg_info["err"],
            "sumw2_bkg_sb": bkg_info["sumw2"],
            "neff_bkg_sb": bkg_info["neff"],
            "nneg_bkg_sb": bkg_info["nneg"],
        })

        if bkg_info["sum"] <= 0.0:
            norm["reason"] = "no_sb_mc_or_nonpositive_yield"
        elif n_data_sb <= 0:
            norm["reason"] = "no_sb_data"
        else:
            alpha, sigma_alpha, valid = alpha_from_yields(
                n_data_sb, bkg_info["sum"], bkg_info["err"]
            )
            norm.update({
                "alpha": alpha,
                "sigma_alpha": sigma_alpha,
                "valid": valid,
                "reason": "ok" if valid else "bad_alpha",
            })
        norms[cat] = norm
    return norms


def empty_result():
    return {
        "S": 0.0, "sigma_S": 0.0,
        "N_signal": 0, "N_signal_negative_weights": 0,
        "N_bkg_sr": 0,
        "Y_bkg_sr": 0.0, "sigma_Y_bkg_sr": 0.0,
        "neff_bkg_sr": 0.0, "nneg_bkg_sr": 0,
        "B": 0.0, "sigma_B": 0.0,
        "sigma_B_from_alpha": 0.0, "sigma_B_from_SR_MC": 0.0,
        "Z": 0.0, "sigma_Z": 0.0,
        "valid": False,
    }


def evaluate_bins(sig, bkg, norms, c_tth, c_vh):
    """Evaluate category-wise signal and alpha-scaled background in SR."""
    sig_cat = assign_cat(sig.score, c_tth, c_vh)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh)
    sig_sr_mask = is_sr_tag(sig.tag)
    bkg_sr_mask = is_sr_tag(bkg.tag)

    results = {}
    for ci, cat in enumerate(CATS):
        norm = norms[cat]
        result = empty_result()
        sig_info = yield_info(sig.weight[sig_sr_mask & (sig_cat == ci)])
        bkg_info = yield_info(bkg.weight[bkg_sr_mask & (bkg_cat == ci)])

        result.update({
            "S": sig_info["sum"],
            "sigma_S": sig_info["err"],
            "N_signal": sig_info["n"],
            "N_signal_negative_weights": sig_info["nneg"],
            "N_bkg_sr": bkg_info["n"],
            "Y_bkg_sr": bkg_info["sum"],
            "sigma_Y_bkg_sr": bkg_info["err"],
            "neff_bkg_sr": bkg_info["neff"],
            "nneg_bkg_sr": bkg_info["nneg"],
        })

        if norm["valid"]:
            stats = bin_significance(
                sig_info["sum"], sig_info["err"],
                bkg_info["sum"], bkg_info["err"],
                norm["alpha"], norm["sigma_alpha"],
            )
            result.update(stats)
        results[cat] = result
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


def group_categories_pass_floor(results, group):
    """Require a measurable alpha and B > MIN_BKG_YIELD for both group cats.

    Categories outside the group cannot move when only this cut changes, so
    they must not veto an individual trial. They are checked once, with the
    final cuts, by categories_pass_floor().
    """
    for cat in GROUP_CATS[group]:
        result = results[cat]
        if not result["valid"]:
            return False
        if not np.isfinite(result["B"]) or result["B"] <= MIN_BKG_YIELD:
            return False
    return True


# =============================================================================
# 6. Raw test-parity sums and pooled final result
# =============================================================================

TEST_SUM_KEYS = (
    "n_sig_sr", "sig_sr_sumw", "sig_sr_sumw2", "nneg_sig_sr",
    "n_bkg_sr", "bkg_sr_sumw", "bkg_sr_sumw2", "nneg_bkg_sr",
    "n_bkg_sb", "bkg_sb_sumw", "bkg_sb_sumw2", "nneg_bkg_sb",
    "n_data_sb",
)


def category_test_sums(sig, bkg, data_sb, c_tth, c_vh):
    """Raw per-category sums for one test parity; deliberately no alpha or Z."""
    sig_cat = assign_cat(sig.score, c_tth, c_vh)
    bkg_cat = assign_cat(bkg.score, c_tth, c_vh)
    data_cat = assign_cat(data_sb.score, c_tth, c_vh)

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
    """Add raw sums from a disjoint test parity into the pooled accumulator."""
    for cat, values in sums.items():
        target = accumulator.setdefault(cat, {key: 0.0 for key in TEST_SUM_KEYS})
        for key, value in values.items():
            target[key] += float(value)
    return accumulator


def pooled_summary(accumulator):
    """Compute one alpha/B/Z per category after pooling all four test parities."""
    norms, results = {}, {}
    for cat in CATS:
        t = accumulator.get(cat, {key: 0.0 for key in TEST_SUM_KEYS})
        n_data_sb = int(t["n_data_sb"])
        y_bkg_sb = t["bkg_sb_sumw"]
        sigma_y_bkg_sb = float(np.sqrt(max(t["bkg_sb_sumw2"], 0.0)))
        neff_sb = (y_bkg_sb ** 2 / t["bkg_sb_sumw2"]
                   if t["bkg_sb_sumw2"] > 0.0 else 0.0)

        alpha, sigma_alpha, valid = alpha_from_yields(
            n_data_sb, y_bkg_sb, sigma_y_bkg_sb
        )
        reason = "ok"
        if y_bkg_sb <= 0.0:
            valid, reason = False, "no_sb_mc_or_nonpositive_yield"
        elif n_data_sb <= 0:
            valid, reason = False, "no_sb_data"

        norm = {
            "category": cat,
            "N_data_sb": n_data_sb,
            "N_bkg_sb": int(t["n_bkg_sb"]),
            "Y_bkg_sb": y_bkg_sb,
            "sigma_Y_bkg_sb": sigma_y_bkg_sb,
            "sumw2_bkg_sb": t["bkg_sb_sumw2"],
            "neff_bkg_sb": neff_sb,
            "nneg_bkg_sb": int(t["nneg_bkg_sb"]),
            "alpha": alpha if valid else float("nan"),
            "sigma_alpha": sigma_alpha if valid else float("nan"),
            "valid": bool(valid),
            "reason": reason if valid else reason,
        }
        norms[cat] = norm

        sig_err = float(np.sqrt(max(t["sig_sr_sumw2"], 0.0)))
        y_bkg_sr = t["bkg_sr_sumw"]
        sigma_y_bkg_sr = float(np.sqrt(max(t["bkg_sr_sumw2"], 0.0)))
        neff_sr = (y_bkg_sr ** 2 / t["bkg_sr_sumw2"]
                   if t["bkg_sr_sumw2"] > 0.0 else 0.0)

        result = empty_result()
        result.update({
            "S": t["sig_sr_sumw"],
            "sigma_S": sig_err,
            "N_signal": int(t["n_sig_sr"]),
            "N_signal_negative_weights": int(t["nneg_sig_sr"]),
            "N_bkg_sr": int(t["n_bkg_sr"]),
            "Y_bkg_sr": y_bkg_sr,
            "sigma_Y_bkg_sr": sigma_y_bkg_sr,
            "neff_bkg_sr": neff_sr,
            "nneg_bkg_sr": int(t["nneg_bkg_sr"]),
        })

        if valid:
            stats = bin_significance(
                result["S"], result["sigma_S"],
                y_bkg_sr, sigma_y_bkg_sr,
                alpha, sigma_alpha,
            )
            result.update(stats)
        results[cat] = result

    invalid_required = [cat for cat in REQUIRED_CATS if not norms[cat]["valid"]]
    if invalid_required:
        details = ", ".join(
            f"{cat}: {norms[cat]['reason']} "
            f"(N_Data_SB={norms[cat]['N_data_sb']}, Y_Bkg_SB={norms[cat]['Y_bkg_sb']:.4g})"
            for cat in invalid_required
        )
        raise RuntimeError(f"Required pooled test categories have no measurable alpha: {details}")

    return norms, results


# =============================================================================
# 7. Diagnostics and printing
# =============================================================================

def print_loaded_files(title, infos):
    print(f"\n{title} file coverage:")
    for process, info in infos.items():
        missing = ",".join(info["missing_eras"]) if info["missing_eras"] else "-"
        print(f"  {process:<34s} files={info['files']:2d} missing_eras={missing}")


def print_process_diagnostics(process_sets, parity, label):
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


def print_normalization(title, norms, c_tth, c_vh):
    print(f"\n{title} — per-category Bkg-MC normalisation")
    print(f"  cuts: c_ttH={c_tth:.5f}, c_VH={c_vh:.5f}")
    header = (
        f"{'Category':12s}{'N_Data_SB':>12s}{'N_Bkg_SB':>11s}"
        f"{'Y_Bkg_SB':>14s}{'sigY_SB':>12s}{'Neff_SB':>11s}"
        f"{'alpha':>13s}{'sig(alpha)':>13s}{'status':>28s}"
    )
    print(header)
    print("-" * len(header))
    for cat in CATS:
        n = norms[cat]
        status = "ok" if n["valid"] else f"SKIP: {n['reason']}"
        print(
            f"{cat:12s}{n['N_data_sb']:12d}{n['N_bkg_sb']:11d}"
            f"{n['Y_bkg_sb']:14.4f}{n['sigma_Y_bkg_sb']:12.4f}"
            f"{n['neff_bkg_sb']:11.2f}"
            f"{n['alpha']:13.6f}{n['sigma_alpha']:13.6f}{status:>28s}"
        )
    print("-" * len(header))


def print_bins(title, results, norms, c_tth=None, c_vh=None):
    z_total, sigma_z_total = combined_z(results)
    print("\n" + "=" * 150)
    if c_tth is None:
        print(title)
    else:
        print(f"{title}: c_ttH={c_tth:.5f}, c_VH={c_vh:.5f}")
    print("=" * 150)
    header = (
        f"{'Category':12s}{'alpha':>11s}"
        f"{'S':>11s}{'sigS':>10s}"
        f"{'N_BkgSR':>10s}{'Y_BkgSR':>13s}{'sigY_BkgSR':>13s}"
        f"{'B':>12s}{'sigB':>11s}{'Z':>11s}{'sigZ':>11s}{'status':>12s}"
    )
    print(header)
    print("-" * len(header))
    for cat in CATS:
        r, n = results[cat], norms[cat]
        status = "used" if r["valid"] else "SKIPPED"
        print(
            f"{cat:12s}{n['alpha']:11.5f}"
            f"{r['S']:11.4f}{r['sigma_S']:10.4f}"
            f"{r['N_bkg_sr']:10d}{r['Y_bkg_sr']:13.4f}{r['sigma_Y_bkg_sr']:13.4f}"
            f"{r['B']:12.4f}{r['sigma_B']:11.4f}"
            f"{r['Z']:11.5f}{r['sigma_Z']:11.5f}{status:>12s}"
        )
    print("-" * len(header))
    used = sum(1 for r in results.values() if r["Z"] > 0.0)
    print(f"Combined Z = {z_total:.6f} ± {sigma_z_total:.6f} ({used}/{len(CATS)} categories contribute)")
    return z_total, sigma_z_total


def print_test_sums(title, sums):
    print(f"\n{title} — raw yields only (no per-fold alpha/Z)")
    header = (
        f"{'Category':12s}{'N_S_SR':>10s}{'S_SR':>13s}"
        f"{'N_B_SR':>10s}{'Y_B_SR':>13s}{'N_B_SB':>10s}"
        f"{'Y_B_SB':>13s}{'N_Data_SB':>12s}"
    )
    print(header)
    print("-" * len(header))
    for cat in CATS:
        t = sums[cat]
        print(
            f"{cat:12s}{int(t['n_sig_sr']):10d}{t['sig_sr_sumw']:13.4f}"
            f"{int(t['n_bkg_sr']):10d}{t['bkg_sr_sumw']:13.4f}"
            f"{int(t['n_bkg_sb']):10d}{t['bkg_sb_sumw']:13.4f}"
            f"{int(t['n_data_sb']):12d}"
        )


def optimize_bayesian(sig_scan, bkg_scan, data_sb_scan, label="",
                      require_all=False):
    """Legacy 2-D joint optimisation of c_ttH and c_VH.

    main() now calls optimize_cut_separate() twice instead, because the two
    cuts are separable and the joint objective surface is nearly flat.
    """
    if optuna is None:
        raise RuntimeError("Optuna is not available; use --no-optuna with the two cut values.")

    def objective(trial):
        c_tth = trial.suggest_float("c_tth", CUT_MIN, CUT_MAX)
        c_vh = trial.suggest_float("c_vh", CUT_MIN, CUT_MAX)

        # Category membership changes with the cuts, so recompute alpha each trial.
        norms = compute_normalization(
            data_sb_scan, bkg_scan, c_tth, c_vh
        )
        results = evaluate_bins(sig_scan, bkg_scan, norms, c_tth, c_vh)
        if not categories_pass_floor(results, require_all=require_all):
            return -1.0
        z, _ = combined_z(results)
        return z if np.isfinite(z) and z > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nBayesian optimisation {label}: {N_TRIALS} trials (2 cuts jointly)...")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None and trial.value > 0.0
    ]
    if not completed:
        raise RuntimeError(
            f"No valid Optuna trial for {label}. Check category-wise Data SB/Bkg MC SB yields "
            "or run --diagnostic-only first."
        )
    best = max(completed, key=lambda trial: trial.value)
    return float(best.params["c_tth"]), float(best.params["c_vh"]), study


def optimize_cut_separate(sig_scan, bkg_scan, data_sb_scan, group,
                          fixed_other_cut, label=""):
    """Optimise one threshold while the other is held fixed.

    group == "ttH" varies c_ttH and maximises
        Z_ttH = sqrt(Z(ttH_high)^2 + Z(ttH_low)^2),
    group == "VH" varies c_VH and maximises the analogous Z_VH. Only the two
    categories of the group are required to have a measurable alpha and
    B > MIN_BKG_YIELD. The fixed threshold touches only the other signal group
    and Other, so one 1-D scan per group is sufficient.
    """
    if optuna is None:
        raise RuntimeError(
            "Optuna is not available; use --no-optuna with the two cut values."
        )
    if group not in GROUP_CATS:
        raise ValueError(f"Unknown optimisation group {group!r}")

    cut_name = "c_tth" if group == "ttH" else "c_vh"
    group_labels = GROUP_CATS[group]

    def objective(trial):
        c = trial.suggest_float(cut_name, CUT_MIN, CUT_MAX)
        if group == "ttH":
            c_tth, c_vh = c, fixed_other_cut
        else:
            c_tth, c_vh = fixed_other_cut, c

        # Category membership changes with the cut, so alpha and B change too.
        norms = compute_normalization(data_sb_scan, bkg_scan, c_tth, c_vh)
        results = evaluate_bins(sig_scan, bkg_scan, norms, c_tth, c_vh)
        if not group_categories_pass_floor(results, group):
            return -1.0
        z_group, _ = group_combined_z(results, group)
        return z_group if np.isfinite(z_group) and z_group > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(
        f"\nBayesian optimisation {label}: {N_TRIALS} trials "
        f"(1 cut: {cut_name}; other fixed at {fixed_other_cut:.5f}; "
        f"require B > {MIN_BKG_YIELD} in {group_labels})..."
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None and trial.value > 0.0
    ]
    if not completed:
        raise RuntimeError(
            f"No valid Optuna trial for {label}. Check category-wise Data SB/Bkg MC SB "
            "yields or run --diagnostic-only first."
        )
    best = max(completed, key=lambda trial: trial.value)
    return float(best.params[cut_name]), study


# =============================================================================
# 8. Main
# =============================================================================

def main():
    global N_TRIALS, MC_SCORED_BASE, DATA_SCORED_BASE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip optimisation and use the supplied two cut values.")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Print scan/test diagnostics with fixed cuts; do not optimise or calculate final Z.")
    parser.add_argument("--require-all-categories", action="store_true",
                        help="Reject a cut pair unless all five categories have measurable alpha and B > MIN_BKG_YIELD.")
    parser.add_argument("--c-tth", type=float, default=0.3)
    parser.add_argument("--c-vh", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--weight-branch", type=str, default=DEFAULT_WEIGHT_BRANCH,
                        choices=["weight_lumi_avg", "weight_lumi"],
                        help=f"MC weight branch (default: {DEFAULT_WEIGHT_BRANCH}).")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR}).")
    parser.add_argument("--scored-tag", type=str, default="",
                        help="Tag inserted into the scored-input directory names, e.g. "
                             "'_minloss' -> ..._4fold_minloss_fold{k} / ..._4fold_data_minloss_fold{k}.")
    args = parser.parse_args()
    N_TRIALS = args.trials

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    if args.scored_tag:
        MC_SCORED_BASE = REPO / (
            f"root_p2root_v3_pt_only_scored_ttH_4class_4fold{args.scored_tag}_fold")
        DATA_SCORED_BASE = REPO / (
            f"root_p2root_v3_pt_only_scored_ttH_4class_4fold_data{args.scored_tag}_fold")
        print(f"MC   scored inputs: {MC_SCORED_BASE}{{0..3}}")
        print(f"Data scored inputs: {DATA_SCORED_BASE}{{0..3}}")

    if not (0.0 <= args.c_tth <= 1.0 and 0.0 <= args.c_vh <= 1.0):
        parser.error("Both cut values must be between 0 and 1.")

    print("=" * 100)
    print("5-CATEGORY SCHEME: train+validation cut scan, pooled-test alpha and Z")
    print("=" * 100)
    print(f"Signal processes: {SIGNAL_DIRS}")
    print(f"Background processes: {BKG_DIRS}")
    print(f"MC weight branch: {args.weight_branch}")
    print(f"Categories: {CATS}")
    print("  ttH_high = argmax ttH and p_ttH > c_ttH")
    print("  ttH_low  = argmax ttH and p_ttH <= c_ttH")
    print("  VH_high  = argmax VH and p_VH > c_VH")
    print("  VH_low   = argmax VH and p_VH <= c_VH")
    print("  Other    = everything else (argmax bkg, argmax ggFVBF, invalid)")
    print("SR: source_tree_tag even (m_llg_refit already in 120-130 GeV)")
    print("SB: source_tree_tag odd ([110,120) or (130,180) GeV), used for alpha only")
    print("No additional mass-window cut is applied; CR Data is not used.")
    print("Cut scan: three parities excluding the held-out test parity (train+validation).")
    print("c_ttH and c_VH are optimised independently (1-D each):")
    print("  c_ttH -> max Z_ttH = sqrt(Z(ttH_high)^2 + Z(ttH_low)^2)")
    print("  c_VH  -> max Z_VH  = sqrt(Z(VH_high)^2  + Z(VH_low)^2);  Other is unaffected")
    print(f"Both categories being scanned must satisfy B > {MIN_BKG_YIELD}.")
    print("Final result: pool four disjoint test parities, then calculate one alpha per category.")
    print(f"Categories required to have measurable alpha: {REQUIRED_CATS}")
    print("Any other category with an unmeasurable alpha is skipped (Z = 0).")
    print("WARNING: verify that DYG mass-point samples and DYJetsToLL_Stitch do not overlap.")

    fold_results = {}
    pooled_sums = {}

    for fold_idx in range(4):
        val_parity = VALID_PARITY[fold_idx]
        test_parity = TEST_PARITY[fold_idx]

        print("\n" + "=" * 90)
        print(f"FOLD {fold_idx}: validation parity={val_parity}, test parity={test_parity}")
        print("=" * 90)

        print("\nLoading all Higgs signal processes...")
        sig_all, sig_infos = load_signal_all(fold_idx, args.weight_branch)
        print(f"  Total Signal MC events loaded: {sig_all.n}")
        print_loaded_files("Signal", sig_infos)

        print("\nLoading all seven Background MC processes...")
        bkg_all, bkg_process_sets, bkg_infos = load_background_all(fold_idx, args.weight_branch)
        print(f"  Total Background MC events loaded: {bkg_all.n}")
        print_loaded_files("Background", bkg_infos)

        print("\nLoading nominal Data sideband with GNN scores...")
        data_sb_all, data_info = load_data_sideband(fold_idx)
        print(f"  Data SB events loaded: {data_sb_all.n}; files={data_info['files']}; "
              f"missing eras={data_info['missing_eras']}")

        # The scan contains all parities except this fold's held-out test parity.
        scan_parities = [p for p in range(4) if p != test_parity]
        sig_scan = select_parity_set(sig_all, scan_parities)
        bkg_scan = select_parity_set(bkg_all, scan_parities)
        data_sb_scan = select_parity_set(data_sb_all, scan_parities)

        # Keep validation and test subsets for transparent diagnostics.
        sig_val = select_parity(sig_all, val_parity)
        sig_test = select_parity(sig_all, test_parity)
        bkg_val = select_parity(bkg_all, val_parity)
        bkg_test = select_parity(bkg_all, test_parity)
        data_sb_val = select_parity(data_sb_all, val_parity)
        data_sb_test = select_parity(data_sb_all, test_parity)

        print(f"\nScan events (parities {scan_parities} = train+validation): "
              f"Signal={sig_scan.n}, BkgMC={bkg_scan.n}, DataSB={data_sb_scan.n}")
        print(f"Validation-only events (parity {val_parity}): Signal={sig_val.n}, "
              f"BkgMC={bkg_val.n}, DataSB={data_sb_val.n}")
        print(f"Test events (parity {test_parity}): Signal={sig_test.n}, "
              f"BkgMC={bkg_test.n}, DataSB={data_sb_test.n}")

        print_process_diagnostics(bkg_process_sets, val_parity, "validation")
        print_process_diagnostics(bkg_process_sets, test_parity, "test")

        if args.diagnostic_only:
            diag_norm_scan = compute_normalization(
                data_sb_scan, bkg_scan, args.c_tth, args.c_vh
            )
            print_normalization(f"FOLD {fold_idx} SCAN (train+validation; diagnostic cuts)",
                                diag_norm_scan, args.c_tth, args.c_vh)
            diag_scan_results = evaluate_bins(
                sig_scan, bkg_scan, diag_norm_scan, args.c_tth, args.c_vh
            )
            print_bins(f"FOLD {fold_idx} SCAN DIAGNOSTIC", diag_scan_results,
                       diag_norm_scan, args.c_tth, args.c_vh)
            diag_test_sums = category_test_sums(
                sig_test, bkg_test, data_sb_test, args.c_tth, args.c_vh
            )
            print_test_sums(f"FOLD {fold_idx} TEST PARITY {test_parity} DIAGNOSTIC",
                            diag_test_sums)
            print("\n[diagnostic-only] No threshold optimisation or final pooled-test Z was run.")
            continue

        studies = {}
        if args.no_optuna:
            c_tth, c_vh = args.c_tth, args.c_vh
            print(f"\n[no-optuna] Using fixed cuts: ttH={c_tth}, VH={c_vh}")
        else:
            # The two cuts are separable because Other only contains
            # bkg/ggFVBF-argmax events and each cut only re-partitions its own
            # argmax group. Optimise one 1-D objective per group.
            c_tth, studies["ttH"] = optimize_cut_separate(
                sig_scan, bkg_scan, data_sb_scan, "ttH", args.c_vh,
                label=f"fold{fold_idx} train+validation ttH group",
            )
            c_vh, studies["VH"] = optimize_cut_separate(
                sig_scan, bkg_scan, data_sb_scan, "VH", c_tth,
                label=f"fold{fold_idx} train+validation VH group",
            )

        # Recompute scan alpha/results with the selected cuts. These are only
        # optimisation diagnostics; the final result is computed on pooled test.
        norm_scan = compute_normalization(
            data_sb_scan, bkg_scan, c_tth, c_vh
        )
        scan_results = evaluate_bins(sig_scan, bkg_scan, norm_scan, c_tth, c_vh)
        if not categories_pass_floor(scan_results, require_all=args.require_all_categories):
            raise RuntimeError(
                f"Fold {fold_idx}: selected cuts fail the scan-sample requirements "
                f"(required alpha categories={REQUIRED_CATS}, B>{MIN_BKG_YIELD})."
            )
        print_normalization(f"FOLD {fold_idx} SCAN (train+validation)",
                            norm_scan, c_tth, c_vh)
        z_scan, sigma_z_scan = print_bins(
            f"FOLD {fold_idx} SCAN (train+validation; optimisation sample)",
            scan_results, norm_scan, c_tth, c_vh,
        )
        z_tth_scan, _ = group_combined_z(scan_results, "ttH")
        z_vh_scan, _ = group_combined_z(scan_results, "VH")
        print(f"  optimisation decomposition: Z_ttH={z_tth_scan:.5f} "
              f"(ttH_high + ttH_low), Z_VH={z_vh_scan:.5f} "
              f"(VH_high + VH_low), Z_Other={scan_results['Other']['Z']:.5f}")

        # Apply this fold's cuts to its test parity. Keep only raw additive sums:
        # no alpha and no Z is evaluated on an individual test fold.
        test_sums = category_test_sums(
            sig_test, bkg_test, data_sb_test, c_tth, c_vh
        )
        add_category_sums(pooled_sums, test_sums)
        print_test_sums(f"FOLD {fold_idx} TEST (parity {test_parity})", test_sums)

        fold_results[fold_idx] = {
            "c_tth": c_tth,
            "c_vh": c_vh,
            "norm_scan": norm_scan,
            "scan_results": scan_results,
            "Z_scan": z_scan,
            "sigma_Z_scan": sigma_z_scan,
            "test_sums": test_sums,
            "test_parity": test_parity,
            "scan_parities": scan_parities,
        }

    if args.diagnostic_only:
        print("\nDiagnostic completed. No optimisation or final pooled-test Z was run.")
        print(f"Log: {log_path}")
        print(f"Output directory: {out_dir}")
        sys.stdout = original_stdout
        log_file.close()
        return

    if len(fold_results) != 4:
        raise RuntimeError(f"Expected four completed folds, got {len(fold_results)}")

    # The four test parities are disjoint. Pool raw sums first, then measure
    # exactly one alpha per category and calculate the final Z once.
    pooled_norms, pooled_results = pooled_summary(pooled_sums)
    z_final, sigma_final = print_bins(
        "POOLED TEST SAMPLE (4 disjoint test parities; final result)",
        pooled_results, pooled_norms,
    )
    z_tth_final, _ = group_combined_z(pooled_results, "ttH")
    z_vh_final, _ = group_combined_z(pooled_results, "VH")
    print(f"  decomposition: Z_ttH={z_tth_final:.5f}, Z_VH={z_vh_final:.5f}, "
          f"Z_Other={pooled_results['Other']['Z']:.5f}")
    below_b_floor = [
        cat for cat, result in pooled_results.items()
        if result["valid"] and result["B"] <= MIN_BKG_YIELD
    ]
    if below_b_floor:
        print(
            "\nWARNING: pooled test category(ies) have B <= "
            f"{MIN_BKG_YIELD:g}: " + ", ".join(
                f"{cat}(B={pooled_results[cat]['B']:.4f})" for cat in below_b_floor
            )
        )

    print("\n" + "=" * 110)
    print("PER-FOLD CUTS AND TEST-PARITY RAW CONTRIBUTIONS")
    print("=" * 110)
    header = (
        f"{'Fold':<7}{'test':>7}{'c_ttH':>10}{'c_VH':>10}{'Z_scan':>12}"
        f"{'S_test':>13}{'Y_BkgSR_test':>16}{'N_DataSB_test':>16}"
    )
    print(header)
    print("-" * len(header))
    for fold_idx, result in fold_results.items():
        ts = result["test_sums"]
        print(
            f"{fold_idx:<7d}{result['test_parity']:7d}"
            f"{result['c_tth']:10.5f}{result['c_vh']:10.5f}{result['Z_scan']:12.5f}"
            f"{sum(ts[c]['sig_sr_sumw'] for c in CATS):13.4f}"
            f"{sum(ts[c]['bkg_sr_sumw'] for c in CATS):16.4f}"
            f"{int(sum(ts[c]['n_data_sb'] for c in CATS)):16d}"
        )
    print("-" * len(header))
    print("Z_scan is only the cut-scan figure of merit on train+validation; it is not the final result.")
    print("The final alpha and Z are calculated once, after pooling the four held-out test parities.")
    print(f"\nPOOLED TEST Z = {z_final:.6f} ± {sigma_final:.6f}")
    print("  per category: " + ", ".join(f"{cat}={pooled_results[cat]['Z']:.4f}" for cat in CATS))

    report_path = out_dir / "final_z.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("5-category HZgamma significance: train+validation scan, pooled test result\n")
        report.write(f"Weight branch: {args.weight_branch}\n")
        report.write("SR: source_tree_tag even; SB: source_tree_tag odd (sideband used for alpha only)\n")
        report.write("Categories: ttH_high / ttH_low / VH_high / VH_low / Other; ggFVBF argmax is included in Other.\n")
        report.write("Procedure\n")
        report.write("  For each fold, scan c_ttH and c_VH separately on the three parities excluding that fold's test parity.\n")
        report.write("  c_ttH maximises Z_ttH = sqrt(Z(ttH_high)^2 + Z(ttH_low)^2);\n")
        report.write("  c_VH  maximises Z_VH  = sqrt(Z(VH_high)^2  + Z(VH_low)^2);  Other is unaffected by both cuts.\n")
        report.write(f"  Both categories in the scanned group must have B > {MIN_BKG_YIELD}.\n")
        report.write("  Apply the selected cuts to the held-out test parity only to assign categories and save raw sums.\n")
        report.write("  Do not calculate alpha or Z per test fold. Pool the four disjoint test parities first.\n")
        report.write("  alpha_c = N(Data SB,c) / Y(BkgMC SB,c); B_c = alpha_c * Y(BkgMC SR,c).\n")
        report.write("  Z_total = sqrt(sum_c Z_c^2), with one independent alpha per category.\n\n")
        report.write("Error propagation\n")
        report.write("  sigma_alpha/alpha = sqrt(1/N_Data_SB + (sigma_Y_Bkg_SB/Y_Bkg_SB)^2).\n")
        report.write("  sigma_Y^2 = sum(w^2); sigma_B^2 = (Y_SR*sigma_alpha)^2 + (alpha*sigma_Y_SR)^2.\n")
        report.write("  sigma_Z propagates sigma_S and sigma_B through the Asimov formula.\n")
        report.write("  sigma_Z_total combines the independent category errors in quadrature.\n\n")
        report.write("Per-fold thresholds and scan figures of merit\n")
        report.write("fold  test_parity  c_ttH  c_VH  Z_scan\n")
        for fold_idx, result in fold_results.items():
            report.write(f"{fold_idx:4d}  {result['test_parity']:11d}  {result['c_tth']:.6f}  "
                         f"{result['c_vh']:.6f}  {result['Z_scan']:.6f}\n")
        report.write(f"\nPOOLED TEST Z = {z_final:.6f} +/- {sigma_final:.6f}\n")
        report.write(
            f"  decomposition: Z_ttH={z_tth_final:.6f}, Z_VH={z_vh_final:.6f}, "
            f"Z_Other={pooled_results['Other']['Z']:.6f}\n"
        )
        report.write("\nPOOLED CATEGORY RESULTS\n")
        for cat in CATS:
            n, r = pooled_norms[cat], pooled_results[cat]
            status = "used" if r["valid"] else f"SKIPPED ({n['reason']})"
            report.write(
                f"{cat:10s} [{status}] N_Data_SB={n['N_data_sb']} "
                f"Y_Bkg_SB={n['Y_bkg_sb']:.6f} +/- {n['sigma_Y_bkg_sb']:.6f} "
                f"Neff_SB={n['neff_bkg_sb']:.2f} alpha={n['alpha']:.6f} +/- {n['sigma_alpha']:.6f} "
                f"S={r['S']:.6f} +/- {r['sigma_S']:.6f} "
                f"Y_Bkg_SR={r['Y_bkg_sr']:.6f} +/- {r['sigma_Y_bkg_sr']:.6f} "
                f"B={r['B']:.6f} +/- {r['sigma_B']:.6f} "
                f"Z={r['Z']:.6f} +/- {r['sigma_Z']:.6f}\n"
            )
        report.write("\nPER-FOLD TRAIN+VALIDATION SCAN CATEGORY RESULTS (diagnostic only)\n")
        for fold_idx, result in fold_results.items():
            report.write(f"FOLD {fold_idx}: c_ttH={result['c_tth']:.6f}, c_VH={result['c_vh']:.6f}\n")
            for cat in CATS:
                n, r = result["norm_scan"][cat], result["scan_results"][cat]
                status = "used" if r["valid"] else f"SKIPPED ({n['reason']})"
                report.write(
                    f"  {cat:10s} [{status}] N_Data_SB={n['N_data_sb']} "
                    f"Y_Bkg_SB={n['Y_bkg_sb']:.6f} alpha={n['alpha']:.6f} +/- {n['sigma_alpha']:.6f} "
                    f"S={r['S']:.6f} B={r['B']:.6f} Z={r['Z']:.6f}\n"
                )

    print(f"\nReport: {report_path}")
    print(f"Log: {log_path}")
    print(f"Output directory: {out_dir}")

    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
