#!/usr/bin/env python3
"""
4-fold OOF significance study — KDE-smoothed Bayesian boundary optimisation.

Same data / folds / cut-based path as significance_4fold.py, but the GNN
boundary search replaces the raw event-counting objective with a
KDE-smoothed continuous estimate of S and B in each (c1, c2) category.

Why KDE?
--------
At c2 ~ 0.99 the high-category MC statistics are tiny (tens of events),
so the raw count histogram is spiky: a single heavy-weight background
event can sit in a score bin and dominate the category yield. The
optimiser then latches onto a statistical hole in the validation score
distribution and calls it a "boundary". Applying that boundary to the
test fold can blow up.

KDE builds a continuous density f_S(s) and f_B(s) from the weighted
events; the category yield is the integral of (weight-density) over the
score window. Because the density is smooth, two nearby score cuts give
nearby yields, so the optimiser cannot exploit single-bin holes.

Implementation
--------------
- Build weighted Gaussian KDE on logit(score) for S and B separately.
  logit avoids the hard walls at 0 and 1 and gives finite bandwidth at
  both ends of the score range.
- The KDE bandwidth H is a free parameter (per-fold, per-sample) but is
  chosen by a robust rule of thumb:

      h = Silverman * max(1, sqrt(Neff) / sqrt(N))

  so heavy weights (small Neff) inflate the bandwidth automatically.
- Category yield for a window (a, b]:

      Y = total_weight * \int_a^b f(s) ds / \int_{lo}^{hi} f(s) ds

  i.e. renormalise the KDE to the total weighted yield and integrate.
  This keeps yields comparable to the raw-count numbers while smoothing
  the boundary dependence.
- The final reported S/B/Z for the chosen (c1, c2) are recomputed with
  the RAW event counting (evaluate_3cat) so the printed numbers are the
  true test-fold yields, not the KDE estimates. KDE is only used to
  drive the optimiser.
"""

import sys
from pathlib import Path

import numpy as np
import optuna
import uproot
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================

OUT_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results_4fold_pt_only_kde"
)

FOLD_DIRS = [
    Path(
        f"/scratchfs2/cms/huangsuyun/GNN4Colliders"
        f"/root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{i}"
    )
    for i in range(4)
]

DATA_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders"
    "/root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
) / "Data"

TEST_PARITY  = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

TREE       = "merged_SR_Sideband"
WEIGHT     = "weight_lumi"
WEIGHT_AVG = "weight_lumi_avg"
MASS       = "m_llg"
SCORE      = "ttH_gnn_score"
IS_SR      = "is_SR"
TREE_TAG   = "source_tree_tag"
FOLDVAR    = "photon_eta_fold"

BRANCHES = [WEIGHT, WEIGHT_AVG, MASS, SCORE, IS_SR, TREE_TAG, FOLDVAR]
DATA_BRANCHES = [WEIGHT, SCORE, IS_SR, TREE_TAG]

MASS_LO = 120.0
MASS_HI = 130.0

SR_TAG_MAP = {
    0: "ttHhad",
    2: "ttHlep",
    4: "untag",
    6: "VH3l",
    8: "VHpTmiss",
}

VALID_SR_TAGS  = np.array(list(SR_TAG_MAP.keys()))
VALID_SB_TAGS  = VALID_SR_TAGS + 1
VALID_FULL_TAGS = np.concatenate([VALID_SR_TAGS, VALID_SB_TAGS])

RUN3 = [
    "2022preEE", "2022postEE",
    "2023preBPix", "2023postBPix",
    "2024", "2025",
]

SIGNAL_GROUP = ("Signal/ttH", RUN3)
BACKGROUND_GROUPS = [
    ("Signal/VH",               RUN3),
    ("Signal/ggF",              RUN3),
    ("Signal/VBF",              RUN3),
    ("Bkg_MC/TT",               RUN3),
    ("Bkg_MC/DYGto2LG_10to100", RUN3),
]

# --- GNN boundary optimisation ---
C1_MIN, C1_MAX = 0.10, 0.95
C2_MIN, C2_MAX = 0.20, 0.999
MIN_GAP        = 0.001

N_TRIALS         = 1000
N_STARTUP_TRIALS = 50
RANDOM_STATE     = 12345

MIN_B_NEFF             = 0.0
MIN_HIGH_SR_TOTAL_YIELD = 0.0
MIN_HIGH_DATA_SB_YIELD  = 0

# --- KDE ---
# Work in logit space so the density doesn't pile up against the 0/1 walls.
EPS = 1e-6
SCORE_LO = EPS
SCORE_HI = 1.0 - EPS
# Integration grid (in score space) for computing category integrals.
N_GRID = 4000
# Floor on the KDE bandwidth, expressed in logit-space standard deviations,
# so a tiny-Neff sample does not collapse to a delta.
MIN_BW_SCALE = 1.0
# Optional: require the KDE high-category background yield to be at least
# this fraction of the total background yield, to avoid chasing holes.
MIN_HIGH_B_FRAC = 0.0


# =============================================================================
# 2. Statistics
# =============================================================================

def asimov_z(s, b):
    if s <= 0.0 or b <= 0.0:
        return 0.0
    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def yield_info(weights):
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return {"sum": 0.0, "err": 0.0, "n": 0, "nneg": 0, "neff": 0.0}
    sumw  = float(np.sum(w))
    sumw2 = float(np.sum(w * w))
    neff  = (sumw * sumw / sumw2) if sumw2 > 0.0 else 0.0
    return {
        "sum":  sumw,
        "err":  float(np.sqrt(sumw2)),
        "n":    int(w.size),
        "nneg": int(np.sum(w < 0.0)),
        "neff": float(neff),
    }


def z_uncertainty(s, b, sigma_s, sigma_b):
    z = asimov_z(s, b)
    if z <= 0.0:
        return 0.0, 0.0, 0.0
    log_term = np.log1p(s / b)
    dz_ds    = log_term / z
    dz_db    = (log_term - s / b) / z
    return (
        float(np.hypot(dz_ds * sigma_s, dz_db * sigma_b)),
        float(abs(dz_ds * sigma_s)),
        float(abs(dz_db * sigma_b)),
    )


def combined_z(results):
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z  = float(np.sqrt(z2))
    if z <= 0.0:
        return 0.0, 0.0
    sigma2 = sum(((r["Z"] / z) * r["sigma_Z"]) ** 2 for r in results.values())
    return z, float(np.sqrt(sigma2))


# =============================================================================
# 3. ROOT helpers  (identical to significance_4fold.py)
# =============================================================================

def empty_arrays():
    return {b: np.array([]) for b in BRANCHES}


def concatenate_arrs(arrays):
    out = {}
    for b in BRANCHES:
        pieces = [a[b] for a in arrays if a[b].size > 0]
        out[b] = np.concatenate(pieces) if pieces else np.array([])
    return out


def apply_mask(arr, mask):
    return {b: arr[b][mask] for b in BRANCHES}


def load_file(path):
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return empty_arrays()
            tree = f[TREE]
            missing = [b for b in BRANCHES if b not in tree.keys()]
            if missing:
                print(f"WARNING: {path} missing branches: {missing}", file=sys.stderr)
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
            pieces.append(arr)
    return concatenate_arrs(pieces)


def load_fold_dir(data_dir):
    print(f"  Loading signal ...")
    signal = load_group(data_dir, SIGNAL_GROUP[0], SIGNAL_GROUP[1])
    background_parts = []
    for subdir, years in BACKGROUND_GROUPS:
        print(f"  Loading {subdir} ...")
        arr = load_group(data_dir, subdir, years)
        background_parts.append((subdir, arr))
    background = concatenate_arrs(
        [a for _, a in background_parts if a[WEIGHT].size > 0]
    )
    return signal, background, background_parts


def load_data_sideband():
    files = sorted(DATA_DIR.rglob("*.root"))
    if not files:
        raise RuntimeError(f"No ROOT files under {DATA_DIR}")
    weights, scores, tags = [], [], []
    for path in files:
        try:
            with uproot.open(str(path)) as f:
                if TREE not in f:
                    continue
                tree = f[TREE]
                arr = tree.arrays(DATA_BRANCHES, library="np")
        except Exception:
            continue
        mask = (arr[IS_SR] == 0) & np.isfinite(arr[WEIGHT]) & np.isfinite(arr[SCORE])
        if np.any(mask):
            weights.append(np.asarray(arr[WEIGHT][mask], dtype=float))
            scores.append(np.asarray(arr[SCORE][mask],  dtype=float))
            tags.append(  np.asarray(arr[TREE_TAG][mask]))
    if not weights:
        raise RuntimeError("No usable Data sideband events (is_SR == 0).")
    return np.concatenate(weights), np.concatenate(scores), np.concatenate(tags)


# =============================================================================
# 4. Event selection
# =============================================================================

def select_sr_parity(arr, parity):
    mask = (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
        & ((arr[FOLDVAR] % 4) == parity)
    )
    return apply_mask(arr, mask)


# =============================================================================
# 5. Raw 3-category evaluation  (unchanged, used for the FINAL test numbers)
# =============================================================================

def evaluate_3cat(
    signal, background,
    data_weights, data_scores,
    c1, c2,
    check_data_sb=False,
):
    results = {}
    for cat in ["high", "medium", "low"]:
        if cat == "high":
            sm = signal[SCORE] > c2
            bm = background[SCORE] > c2
            dm = data_scores > c2
        elif cat == "medium":
            sm = (signal[SCORE] > c1)  & (signal[SCORE] <= c2)
            bm = (background[SCORE] > c1) & (background[SCORE] <= c2)
            dm = (data_scores > c1) & (data_scores <= c2)
        else:
            sm = signal[SCORE] <= c1
            bm = background[SCORE] <= c1
            dm = data_scores <= c1

        S = yield_info(signal[WEIGHT][sm])
        B = yield_info(background[WEIGHT][bm])
        data_sb_yield = float(np.sum(data_weights[dm])) if data_weights.size > 0 else 0.0

        if B["sum"] <= 0.0:
            return {}, 0.0, False
        if B["neff"] < MIN_B_NEFF:
            return {}, 0.0, False
        if cat == "high":
            if S["sum"] + B["sum"] < MIN_HIGH_SR_TOTAL_YIELD:
                return {}, 0.0, False
            if check_data_sb and data_sb_yield <= MIN_HIGH_DATA_SB_YIELD:
                return {}, 0.0, False

        sigma_z, sigma_z_s, sigma_z_b = z_uncertainty(S["sum"], B["sum"], S["err"], B["err"])
        results[cat] = {
            "S": S, "B": B,
            "DataSBYield": data_sb_yield,
            "Z": asimov_z(S["sum"], B["sum"]),
            "sigma_Z": sigma_z,
            "sigma_Z_from_S": sigma_z_s,
            "sigma_Z_from_B": sigma_z_b,
        }

    z_total, _ = combined_z(results)
    return results, z_total, True


def print_3cat(title, results, c1, c2):
    z, sigma_z = combined_z(results)
    w = 120
    print("\n" + "=" * w)
    print(f"{title}: c1={c1:.6f}, c2={c2:.6f}")
    print("=" * w)
    hdr = (
        f"{'Cat':<8}{'S':>10}{'σS':>10}{'B':>12}{'σB':>10}"
        f"{'NeffB':>10}{'NB':>8}{'Nneg':>8}{'DataSBYield':>14}"
        f"{'Z':>10}{'σZ(S)':>10}{'σZ(B)':>10}{'σZ':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cat in ["high", "medium", "low"]:
        r = results[cat]
        S, B = r["S"], r["B"]
        print(
            f"{cat:<8}"
            f"{S['sum']:>10.4f}{S['err']:>10.4f}"
            f"{B['sum']:>12.4f}{B['err']:>10.4f}"
            f"{B['neff']:>10.2f}{B['n']:>8d}{B['nneg']:>8d}"
            f"{r['DataSBYield']:>14.4f}"
            f"{r['Z']:>10.4f}{r['sigma_Z_from_S']:>10.4f}"
            f"{r['sigma_Z_from_B']:>10.4f}{r['sigma_Z']:>10.4f}"
        )
    print("-" * len(hdr))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}")


# =============================================================================
# 6. Cut-based significance  (identical to significance_4fold.py)
# =============================================================================

def calculate_cutbased(signal, background, data_tags):
    results = {}
    for tag, name in SR_TAG_MAP.items():
        sm = signal[TREE_TAG] == tag
        bm = background[TREE_TAG] == tag
        S = yield_info(signal[WEIGHT][sm])
        B = yield_info(background[WEIGHT][bm])
        data_sb_n = int(np.sum(data_tags == tag + 1)) if data_tags.size > 0 else 0
        z = asimov_z(S["sum"], B["sum"])
        sigma_z, sigma_z_s, sigma_z_b = z_uncertainty(S["sum"], B["sum"], S["err"], B["err"])
        results[name] = {
            "S": S, "B": B,
            "DataSB": data_sb_n,
            "Z": z,
            "sigma_Z": sigma_z,
            "sigma_Z_from_S": sigma_z_s,
            "sigma_Z_from_B": sigma_z_b,
        }
    return results


def print_cutbased_table(title, results):
    z, sigma_z = combined_z(results)
    w = 125
    print("\n" + "=" * w)
    print(title)
    print("=" * w)
    hdr = (
        f"{'Cat':<10}{'S':>10}{'σS':>10}{'B':>12}{'σB':>10}"
        f"{'NeffB':>10}{'NB':>8}{'Nneg':>8}{'DataSB':>9}"
        f"{'Z':>10}{'σZ(S)':>10}{'σZ(B)':>10}{'σZ':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in SR_TAG_MAP.values():
        r = results[name]
        S, B = r["S"], r["B"]
        print(
            f"{name:<10}"
            f"{S['sum']:>10.4f}{S['err']:>10.4f}"
            f"{B['sum']:>12.4f}{B['err']:>10.4f}"
            f"{B['neff']:>10.2f}{B['n']:>8d}{B['nneg']:>8d}"
            f"{r['DataSB']:>9d}"
            f"{r['Z']:>10.4f}{r['sigma_Z_from_S']:>10.4f}"
            f"{r['sigma_Z_from_B']:>10.4f}{r['sigma_Z']:>10.4f}"
        )
    print("-" * len(hdr))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}")


# =============================================================================
# 7. KDE helpers
# =============================================================================

def _clip_score(s):
    return np.clip(s, SCORE_LO, SCORE_HI)


def _to_logit(s):
    s = _clip_score(s)
    return np.log(s / (1.0 - s))


def _bandwidth(scores, weights):
    """Robust bandwidth in logit space.

    std(x) is a bad scale estimator here because the background logit
    distribution is bimodal (most events near score 0 => large negative
    logit, a handful near score 1 => large positive logit), so std is
    dominated by the gap between the two modes (~3) and the KDE bandwidth
    comes out wider than a peak, smearing the high-score tail across the
    whole score range. Use the IQR-based Silverman factor instead, which
    tracks the *intra-mode* spread, and do not inflate by N/Neff:
    gaussian_kde already incorporates the weights, so that factor double-
    counts the heavy-weight events and over-smooths further.
    """
    x = _to_logit(scores)
    if x.size < 2:
        return 0.5
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    scale = iqr / 1.349 if iqr > 0 else np.std(x)
    if scale <= 0 or not np.isfinite(scale):
        return 0.5
    neff = max(yield_info(weights)["neff"], 2.0)
    return max(0.9 * scale * neff ** (-1.0 / 5.0), 1e-3)


def build_kde(scores, weights):
    """Weighted Gaussian KDE in logit space.

    Returns (kde, total_yield) or (None, 0.0) if there is nothing to fit.
    """
    scores  = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.size == 0:
        return None, 0.0
    info = yield_info(weights)
    if info["sum"] == 0.0 or scores.size < 2:
        return None, info["sum"]
    x = _to_logit(scores)
    try:
        kde = gaussian_kde(x, weights=weights, bw_method=_bandwidth(scores, weights))
    except Exception:
        kde = gaussian_kde(x, bw_method=_bandwidth(scores, weights))
    return kde, info["sum"]


def build_kde_cdf(kde, grid_x):
    """Precompute the score-space cumulative density on grid_x (logit grid).

    Returns:
      s_grid : score value at each grid point
      cdf    : normalised cumulative integral in score space, monotonically
               rising from ~0 to ~1.

    Once built, the integral over any (s_lo, s_hi] is just a lookup:
        integral = tot * (cdf(idx_hi) - cdf(idx_lo))
    making every optimiser trial O(1) instead of re-evaluating the KDE.
    """
    s_grid = 1.0 / (1.0 + np.exp(-grid_x))
    jac = s_grid * (1.0 - s_grid)
    dx = np.empty_like(grid_x)
    dx[:-1] = np.diff(grid_x)
    dx[-1]  = dx[-2]
    # per-bin score-space integral of the (logit-space) KDE density
    bin_mass = kde(grid_x) * jac * dx
    cdf = np.concatenate([[0.0], np.cumsum(bin_mass)])
    cdf /= cdf[-1] if cdf[-1] > 0.0 else 1.0
    # cdf now has length len(grid_x)+1, aligned with bin edges.
    return s_grid, cdf


def _cdf_lookup(s_grid, cdf, s_cut):
    """Return cdf value just above s_cut via searchsorted (right edge)."""
    idx = np.searchsorted(s_grid, s_cut, side="right")
    idx = min(idx, len(cdf) - 1)
    return cdf[idx]


def category_yield_cdf(s_grid, cdf, total_yield, lo, hi):
    """O(1) category yield via CDF difference."""
    if total_yield <= 0.0:
        return 0.0
    s_lo = _clip_score(lo)
    s_hi = _clip_score(hi)
    if s_hi <= s_lo:
        return 0.0
    return total_yield * (
        _cdf_lookup(s_grid, cdf, s_hi) - _cdf_lookup(s_grid, cdf, s_lo)
    )


# =============================================================================
# 8. KDE-smoothed 3-category objective
# =============================================================================

def kde_3cat_yields(s_grid, cdf_s, tot_s, cdf_b, tot_b, c1, c2):
    s_high = category_yield_cdf(s_grid, cdf_s, tot_s, c2, 1.0)
    s_med  = category_yield_cdf(s_grid, cdf_s, tot_s, c1, c2)
    s_low  = category_yield_cdf(s_grid, cdf_s, tot_s, 0.0, c1)
    b_high = category_yield_cdf(s_grid, cdf_b, tot_b, c2, 1.0)
    b_med  = category_yield_cdf(s_grid, cdf_b, tot_b, c1, c2)
    b_low  = category_yield_cdf(s_grid, cdf_b, tot_b, 0.0, c1)
    return {
        "high":   (s_high, b_high),
        "medium": (s_med,  b_med),
        "low":    (s_low,  b_low),
    }


def kde_objective_value(yields, tot_b):
    """Asimov Z from KDE yields, with the same validity guards as the raw
    evaluator (B>0, optional Neff floor replaced by a yield-floor proxy).
    """
    z2 = 0.0
    for cat in ["high", "medium", "low"]:
        s, b = yields[cat]
        if b <= 0.0:
            return None
        if cat == "high":
            if MIN_HIGH_B_FRAC > 0.0 and b < MIN_HIGH_B_FRAC * tot_b:
                return None
        z2 += asimov_z(s, b) ** 2
    return float(np.sqrt(z2)) if z2 > 0.0 else 0.0


def optimize_kde(
    signal, background,
    data_weights, data_scores,
    check_data_sb=False,
    label="",
):
    sig_scores  = signal[SCORE]
    sig_weights = signal[WEIGHT]
    bkg_scores  = background[SCORE]
    bkg_weights = background[WEIGHT]

    kde_s, tot_s = build_kde(sig_scores, sig_weights)
    kde_b, tot_b = build_kde(bkg_scores, bkg_weights)
    if kde_b is None or tot_b <= 0.0:
        raise RuntimeError(f"Cannot build background KDE for {label}.")

    # Precompute the score-space CDFs ONCE. The expensive KDE evaluation over
    # ~78000 events happens exactly twice per fold here; every optimiser
    # trial after that is an O(1) searchsorted lookup.
    grid_x = np.linspace(_to_logit(SCORE_LO), _to_logit(SCORE_HI), N_GRID)
    s_grid, cdf_s = build_kde_cdf(kde_s, grid_x)
    _,    cdf_b = build_kde_cdf(kde_b, grid_x)

    def objective(trial):
        c1     = trial.suggest_float("c1", C1_MIN, C1_MAX)
        c2_low = max(C2_MIN, c1 + MIN_GAP)
        if c2_low >= C2_MAX:
            return -1.0
        c2 = trial.suggest_float("c2", c2_low, C2_MAX)
        yields = kde_3cat_yields(s_grid, cdf_s, tot_s, cdf_b, tot_b, c1, c2)
        z = kde_objective_value(yields, tot_b)
        trial.set_user_attr("valid", z is not None)
        return z if z is not None else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, n_startup_trials=N_STARTUP_TRIALS
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nKDE Bayesian optimisation {label}: {N_TRIALS} trials ...")
    print(f"  KDE: S_N={sig_scores.size} B_N={bkg_scores.size} "
          f"S_yield={tot_s:.4f} B_yield={tot_b:.4f}")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

    valid_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("valid", False)
    ]
    if not valid_trials:
        raise RuntimeError(f"No valid KDE trial for {label}.")
    best = max(valid_trials, key=lambda t: t.value)
    c1   = float(best.params["c1"])
    c2   = float(best.params["c2"])
    print(f"  KDE best Z(kde)={best.value:.5f} at c1={c1:.6f}, c2={c2:.6f}")
    # Recompute raw yields on the validation set for an honest comparison.
    val_res, _, _ = evaluate_3cat(
        signal, background,
        data_weights, data_scores,
        c1, c2, check_data_sb=check_data_sb,
    )
    return c1, c2, val_res


# =============================================================================
# 9. Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_weights, data_scores, data_tags = load_data_sideband()

    empty_w = np.array([])
    empty_s = np.array([])

    fold_results = {}

    for fold_idx in range(4):
        val_p  = VALID_PARITY[fold_idx]
        test_p = TEST_PARITY[fold_idx]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}  val parity={val_p}  test parity={test_p}")
        print(f"{'='*70}")
        print(f"Loading from {FOLD_DIRS[fold_idx].name} ...")

        signal_all, bkg_all, _ = load_fold_dir(FOLD_DIRS[fold_idx])

        sig_val  = select_sr_parity(signal_all, val_p)
        bkg_val  = select_sr_parity(bkg_all,    val_p)
        sig_test = select_sr_parity(signal_all, test_p)
        bkg_test = select_sr_parity(bkg_all,    test_p)

        print(
            f"Val  set: N_sig={sig_val[WEIGHT].size:6d}, "
            f"N_bkg={bkg_val[WEIGHT].size:6d}, "
            f"S={np.sum(sig_val[WEIGHT]):.4f}, "
            f"B={np.sum(bkg_val[WEIGHT]):.4f}"
        )
        print(
            f"Test set: N_sig={sig_test[WEIGHT].size:6d}, "
            f"N_bkg={bkg_test[WEIGHT].size:6d}, "
            f"S={np.sum(sig_test[WEIGHT]):.4f}, "
            f"B={np.sum(bkg_test[WEIGHT]):.4f}"
        )

        # --- Cut-based on test set ---
        cut_test = calculate_cutbased(sig_test, bkg_test, data_tags)
        print_cutbased_table(f"FOLD {fold_idx} TEST — CUT-BASED", cut_test)
        z_cut_test, _ = combined_z(cut_test)

        # --- KDE Bayesian optimisation on validation set ---
        use_data = (fold_idx == 0)
        dw = data_weights if use_data else empty_w
        ds = data_scores  if use_data else empty_s

        c1_best, c2_best, val_res = optimize_kde(
            sig_val, bkg_val, dw, ds,
            check_data_sb=use_data,
            label=f"fold{fold_idx} val",
        )
        z_val, _ = combined_z(val_res)
        print_3cat(
            f"FOLD {fold_idx} VALIDATION (KDE-selected, raw-count)",
            val_res, c1_best, c2_best,
        )
        print(f"Fold {fold_idx} chosen boundary: c1={c1_best:.6f}, c2={c2_best:.6f}")

        # --- Raw-count evaluation on test set with the KDE-chosen boundary ---
        test_results, z_test, valid = evaluate_3cat(
            sig_test, bkg_test,
            empty_w, empty_s,
            c1_best, c2_best,
            check_data_sb=False,
        )
        if not valid:
            print(
                f"WARNING: fold {fold_idx} GNN test evaluation invalid.",
                file=sys.stderr,
            )
            z_test       = 0.0
            test_results = {}
        if test_results:
            print_3cat(f"FOLD {fold_idx} TEST — GNN (KDE)", test_results, c1_best, c2_best)

        fold_results[fold_idx] = {
            "c1": c1_best, "c2": c2_best,
            "val_results":      val_res,
            "test_results":     test_results,
            "cut_test_results": cut_test,
            "Z_val":      z_val,
            "Z_test":     z_test,
            "Z_cut_test": z_cut_test,
        }

    # -------------------------------------------------------------------------
    # Combined Z
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PER-FOLD SUMMARY")
    print("=" * 80)
    print(
        f"{'Fold':<6}"
        f"{'c1':>10}{'c2':>10}"
        f"{'Z_val':>12}{'Z_GNN_test':>14}{'Z_cut_test':>14}"
    )
    print("-" * 68)

    gnn_z_sq = 0.0
    cut_z_sq = 0.0
    for fold_idx, r in fold_results.items():
        print(
            f"{fold_idx:<6}"
            f"{r['c1']:>10.6f}{r['c2']:>10.6f}"
            f"{r['Z_val']:>12.6f}{r['Z_test']:>14.6f}{r['Z_cut_test']:>14.6f}"
        )
        gnn_z_sq += r["Z_test"] ** 2
        cut_z_sq += r["Z_cut_test"] ** 2

    z_gnn_final = float(np.sqrt(gnn_z_sq))
    z_cut_final = float(np.sqrt(cut_z_sq))

    sigma_sq = 0.0
    for r in fold_results.values():
        if r["test_results"] and z_gnn_final > 0.0:
            _, sigma_k = combined_z(r["test_results"])
            sigma_sq += ((r["Z_test"] / z_gnn_final) * sigma_k) ** 2
    sigma_gnn_final = float(np.sqrt(sigma_sq))

    sigma_sq = 0.0
    for r in fold_results.values():
        if r["cut_test_results"] and z_cut_final > 0.0:
            _, sigma_k = combined_z(r["cut_test_results"])
            sigma_sq += ((r["Z_cut_test"] / z_cut_final) * sigma_k) ** 2
    sigma_cut_final = float(np.sqrt(sigma_sq))

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(
        f"4-fold GNN (KDE) combined Z = {z_gnn_final:.6f} ± {sigma_gnn_final:.6f}"
        f"  = sqrt("
        + " + ".join(f"{r['Z_test']:.4f}^2" for r in fold_results.values())
        + ")"
    )
    print(
        f"4-fold cut-based          Z = {z_cut_final:.6f} ± {sigma_cut_final:.6f}"
        f"  = sqrt("
        + " + ".join(f"{r['Z_cut_test']:.4f}^2" for r in fold_results.values())
        + ")"
    )
    if z_cut_final > 0.0:
        ratio = z_gnn_final / z_cut_final
        print(f"GNN (KDE) / cut-based = {ratio:.4f} ({(ratio - 1.0) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
