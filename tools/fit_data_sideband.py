#!/usr/bin/env python3
"""
Method 1: Data sideband mass fit for the GNN high category.

Purpose
-------
Keep the original Bayesian/Optuna significance code unchanged.
This is an ADDITIONAL script for Method 1:

  1. Select Data sideband events in the GNN high category.
  2. Fit m_llg in the sidebands with:
       - ExpPoly2
       - Bern3
  3. Extrapolate each fitted shape into 120--130 GeV.
  4. Use the extrapolated Data background as B.
  5. Use ttH MC 2-fold OOF as S.
  6. Calculate the Asimov significance Z.

Default:
  GNN high: score > 0.985
  fit sidebands: 110--120 and 130--180 GeV
  signal region: 120--130 GeV

Important
---------
Data is unweighted. The sideband fit is a shape fit to Data event counts.
The signal still uses weight_lumi from the 2-fold OOF ttH MC.
The old MC background is printed only for comparison; it is NOT used in
the new Data-driven Z.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gammaln


# =============================================================================
# 1. Configuration
# =============================================================================

OUT_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results_lypt/sideband_fit"
)

DATA_DIR_FOLD0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
)

DATA_DIR_FOLD1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold1"
)

DATA_DIR = DATA_DIR_FOLD0 / "Data"

TREE = "merged_SR_Sideband"

WEIGHT = "weight_lumi"
MASS = "m_llg"
SCORE = "ttH_gnn_score"
IS_SR = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR = "photon_eta_fold"

DEFAULT_C1 = 0.900
DEFAULT_C2 = 0.985

DEFAULT_FIT_MIN = 110.0
DEFAULT_SIGNAL_LO = 120.0
DEFAULT_SIGNAL_HI = 130.0
DEFAULT_FIT_MAX = 180.0

FIT_BIN_WIDTH = 1.0
PLOT_BIN_WIDTH = 2.0

FOLD0_OOF_PARITY = 0
FOLD1_OOF_PARITY = 1

SR_TAG_MAP = {
    0: "ttHhad",
    2: "ttHlep",
    4: "untag",
    6: "VH3l",
    8: "VHpTmiss",
}

VALID_SR_TAGS = np.array(list(SR_TAG_MAP.keys()), dtype=int)

RUN3 = [
    "2022preEE",
    "2022postEE",
    "2023preBPix",
    "2023postBPix",
    "2024",
    "2025",
]

BACKGROUND_GROUPS = [
    "Signal/VH",
    "Signal/ggF",
    "Signal/VBF",
    "Bkg_MC/TT",
    "Bkg_MC/DYGto2LG_10to100",
]


# =============================================================================
# 2. Statistics
# =============================================================================

def asimov_z(s, b):
    """Asimov significance."""
    if s <= 0.0 or b <= 0.0:
        return 0.0

    x = 2.0 * ((s + b) * np.log1p(s / b) - s)

    return float(np.sqrt(x)) if x > 0.0 else 0.0


def weighted_yield(weights):
    """Weighted yield, MC statistical uncertainty, and raw count."""
    w = np.asarray(weights, dtype=float)

    if w.size == 0:
        return 0.0, 0.0, 0

    return (
        float(np.sum(w)),
        float(np.sqrt(np.sum(w * w))),
        int(w.size),
    )


# =============================================================================
# 3. ROOT helpers
# =============================================================================

def read_tree(path, branches):
    """Read merged_SR_Sideband from one ROOT file."""
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return None

            tree = f[TREE]
            keys = set(tree.keys())

            missing = [b for b in branches if b not in keys]

            if missing:
                print(
                    f"WARNING: {path} missing branches: {missing}",
                    file=sys.stderr,
                )
                return None

            return tree.arrays(branches, library="np")

    except Exception as exc:
        print(
            f"WARNING: cannot read {path}: {exc}",
            file=sys.stderr,
        )
        return None


def load_data():
    """Load all Data ROOT files under fold0/Data."""
    branches = [
        MASS,
        SCORE,
        IS_SR,
        TREE_TAG,
    ]

    files = sorted(DATA_DIR.rglob("*.root"))

    if not files:
        raise RuntimeError(
            f"No ROOT files found under {DATA_DIR}"
        )

    pieces = []

    print("\nLoading Data:")
    print(f"  {DATA_DIR}")
    print(f"  ROOT files found = {len(files)}")

    for path in files:
        arr = read_tree(path, branches)

        if arr is None:
            continue

        pieces.append(arr)

    if not pieces:
        raise RuntimeError("No usable Data trees were loaded.")

    return {
        MASS: np.concatenate([np.asarray(x[MASS]) for x in pieces]),
        SCORE: np.concatenate([np.asarray(x[SCORE]) for x in pieces]),
        IS_SR: np.concatenate([np.asarray(x[IS_SR]) for x in pieces]),
        TREE_TAG: np.concatenate([np.asarray(x[TREE_TAG]) for x in pieces]),
    }


def load_mc_group(data_dir, subdir, years):
    """Load one MC group from all requested years."""
    branches = [
        WEIGHT,
        MASS,
        SCORE,
        IS_SR,
        TREE_TAG,
        FOLDVAR,
    ]

    pieces = []

    for year in years:
        path = data_dir / subdir / f"{year}.root"

        if not path.exists():
            print(f"WARNING: missing {path}")
            continue

        arr = read_tree(path, branches)

        if arr is not None:
            pieces.append(arr)

    if not pieces:
        return {
            b: np.array([])
            for b in branches
        }

    return {
        b: np.concatenate([np.asarray(x[b]) for x in pieces])
        for b in branches
    }


def mc_oof_selection(arr, parity, fit_cfg):
    """Selection for signal/background MC in the 120--130 SR, OOF."""
    return (
        (arr[IS_SR] == 1)
        & (arr[MASS] > fit_cfg["signal_lo"])
        & (arr[MASS] < fit_cfg["signal_hi"])
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & ((arr[FOLDVAR] % 2) == parity)
        & np.isfinite(arr[MASS])
        & np.isfinite(arr[SCORE])
        & np.isfinite(arr[WEIGHT])
    )


def load_tth_signal_oof(fit_cfg):
    """Load ttH and construct true 2-fold OOF signal."""
    print("\nLoading ttH signal for 2-fold OOF...")

    fold0 = load_mc_group(
        DATA_DIR_FOLD0,
        "Signal/ttH",
        RUN3,
    )

    fold1 = load_mc_group(
        DATA_DIR_FOLD1,
        "Signal/ttH",
        RUN3,
    )

    m0 = mc_oof_selection(
        fold0,
        FOLD0_OOF_PARITY,
        fit_cfg,
    )

    m1 = mc_oof_selection(
        fold1,
        FOLD1_OOF_PARITY,
        fit_cfg,
    )

    return {
        MASS: np.concatenate([
            fold0[MASS][m0],
            fold1[MASS][m1],
        ]),
        SCORE: np.concatenate([
            fold0[SCORE][m0],
            fold1[SCORE][m1],
        ]),
        WEIGHT: np.concatenate([
            fold0[WEIGHT][m0],
            fold1[WEIGHT][m1],
        ]),
    }


def load_background_mc_oof(fit_cfg):
    """
    Load old MC background for comparison only.
    """
    all_mass = []
    all_score = []
    all_weight = []

    for subdir in BACKGROUND_GROUPS:
        print(f"Loading background MC: {subdir}")

        fold0 = load_mc_group(
            DATA_DIR_FOLD0,
            subdir,
            RUN3,
        )

        fold1 = load_mc_group(
            DATA_DIR_FOLD1,
            subdir,
            RUN3,
        )

        m0 = mc_oof_selection(
            fold0,
            FOLD0_OOF_PARITY,
            fit_cfg,
        )

        m1 = mc_oof_selection(
            fold1,
            FOLD1_OOF_PARITY,
            fit_cfg,
        )

        all_mass.extend([
            fold0[MASS][m0],
            fold1[MASS][m1],
        ])

        all_score.extend([
            fold0[SCORE][m0],
            fold1[SCORE][m1],
        ])

        all_weight.extend([
            fold0[WEIGHT][m0],
            fold1[WEIGHT][m1],
        ])

    if not all_weight:
        return None

    return {
        MASS: np.concatenate(all_mass),
        SCORE: np.concatenate(all_score),
        WEIGHT: np.concatenate(all_weight),
    }


# =============================================================================
# 4. Data selection
# =============================================================================

def select_data_sideband(data, c2, fit_cfg):
    """
    Select Data:
      - sideband (110--120 and 130--180 by default)
      - is_SR == 0
      - GNN high: score > c2

    Data is unweighted.
    """
    mass = np.asarray(data[MASS], dtype=float)
    score = np.asarray(data[SCORE], dtype=float)

    finite = np.isfinite(mass) & np.isfinite(score)

    left = (
        (mass >= fit_cfg["fit_min"])
        & (mass < fit_cfg["signal_lo"])
    )

    right = (
        (mass > fit_cfg["signal_hi"])
        & (mass <= fit_cfg["fit_max"])
    )

    mask = (
        finite
        & (data[IS_SR] == 0)
        & (score > c2)
        & (left | right)
    )

    return mass[mask]


def select_data_signal_region(data, c2, fit_cfg):
    """
    Data in 120--130 and GNN high.
    Diagnostic only; never used as B.
    """
    mass = np.asarray(data[MASS], dtype=float)
    score = np.asarray(data[SCORE], dtype=float)

    mask = (
        np.isfinite(mass)
        & np.isfinite(score)
        & (data[IS_SR] == 1)
        & (mass > fit_cfg["signal_lo"])
        & (mass < fit_cfg["signal_hi"])
        & (score > c2)
    )

    return mass[mask]


# =============================================================================
# 5. Sideband bins
# =============================================================================

def make_sideband_bins(fit_cfg):
    """Return separate 1-GeV bins for left and right sidebands."""
    left_edges = np.arange(
        fit_cfg["fit_min"],
        fit_cfg["signal_lo"] + FIT_BIN_WIDTH,
        FIT_BIN_WIDTH,
    )

    right_edges = np.arange(
        fit_cfg["signal_hi"],
        fit_cfg["fit_max"] + FIT_BIN_WIDTH,
        FIT_BIN_WIDTH,
    )

    return left_edges, right_edges


def histogram_sidebands(masses, fit_cfg):
    """Histogram Data sideband with 1-GeV bins."""
    left_edges, right_edges = make_sideband_bins(fit_cfg)

    left_mask = (
        (masses >= fit_cfg["fit_min"])
        & (masses < fit_cfg["signal_lo"])
    )

    right_mask = (
        (masses > fit_cfg["signal_hi"])
        & (masses <= fit_cfg["fit_max"])
    )

    left_counts, _ = np.histogram(
        masses[left_mask],
        bins=left_edges,
    )

    right_counts, _ = np.histogram(
        masses[right_mask],
        bins=right_edges,
    )

    return (
        left_edges,
        left_counts,
        right_edges,
        right_counts,
    )


# =============================================================================
# 6. Fit models
# =============================================================================

def exp_poly2_raw(x, params):
    """
    ExpPoly2:
        f(x) = exp(p0 + p1*t + p2*t^2)

    t=(m-145)/35 keeps the numerical scale reasonable.
    """
    p0, p1, p2 = params

    t = (np.asarray(x, dtype=float) - 145.0) / 35.0

    exponent = (
        p0
        + p1 * t
        + p2 * t * t
    )

    exponent = np.clip(
        exponent,
        -50.0,
        50.0,
    )

    return np.exp(exponent)


def bern3_raw(x, params, fit_cfg):
    """
    Positive Bernstein polynomial of degree 3.

    Coefficients are exp(q_i), so the PDF stays positive.
    """
    q0, q1, q2, q3 = params

    x = np.asarray(x, dtype=float)

    t = (
        (x - fit_cfg["fit_min"])
        / (fit_cfg["fit_max"] - fit_cfg["fit_min"])
    )

    t = np.clip(t, 0.0, 1.0)

    coeff = np.exp(
        np.clip(
            np.array([q0, q1, q2, q3]),
            -30.0,
            30.0,
        )
    )

    b0 = (1.0 - t) ** 3
    b1 = 3.0 * t * (1.0 - t) ** 2
    b2 = 3.0 * t * t * (1.0 - t)
    b3 = t ** 3

    return (
        coeff[0] * b0
        + coeff[1] * b1
        + coeff[2] * b2
        + coeff[3] * b3
    )


def model_raw(x, model_name, params, fit_cfg):
    if model_name == "ExpPoly2":
        return exp_poly2_raw(x, params)

    if model_name == "Bern3":
        return bern3_raw(x, params, fit_cfg)

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def numerical_integral(func, lo, hi, n=400):
    """Numerical integral with trapezoidal rule."""
    if hi <= lo:
        return 0.0

    x = np.linspace(
        lo,
        hi,
        n,
    )

    y = func(x)

    value = np.trapz(
        y,
        x,
    )

    return float(value)


def sideband_integral(params, model_name, fit_cfg):
    """Integral of the raw shape over the fitted sidebands."""
    func = lambda x: model_raw(
        x,
        model_name,
        params,
        fit_cfg,
    )

    left = numerical_integral(
        func,
        fit_cfg["fit_min"],
        fit_cfg["signal_lo"],
        n=500,
    )

    right = numerical_integral(
        func,
        fit_cfg["signal_hi"],
        fit_cfg["fit_max"],
        n=2500,
    )

    return left + right


def poisson_nll(params, masses, model_name, fit_cfg):
    """
    Conditional Poisson likelihood for the sideband shape.

    The total observed number of sideband events is fixed to N_SB.
    The fitted function therefore determines the relative shape.

    For each sideband bin:
        mu_i = N_SB * integral_i(f) / integral_SB(f)

    The extrapolated SR background is:
        B_SR = N_SB * integral_SR(f) / integral_SB(f)
    """
    total_norm = sideband_integral(
        params,
        model_name,
        fit_cfg,
    )

    if (
        not np.isfinite(total_norm)
        or total_norm <= 0.0
    ):
        return 1e100

    left_edges, left_counts, right_edges, right_counts = (
        histogram_sidebands(
            masses,
            fit_cfg,
        )
    )

    observed = np.concatenate([
        left_counts,
        right_counts,
    ])

    expected = []

    for lo, hi in zip(
        left_edges[:-1],
        left_edges[1:],
    ):
        func = lambda x: model_raw(
            x,
            model_name,
            params,
            fit_cfg,
        )

        integral = numerical_integral(
            func,
            lo,
            hi,
            n=25,
        )

        expected.append(
            len(masses) * integral / total_norm
        )

    for lo, hi in zip(
        right_edges[:-1],
        right_edges[1:],
    ):
        func = lambda x: model_raw(
            x,
            model_name,
            params,
            fit_cfg,
        )

        integral = numerical_integral(
            func,
            lo,
            hi,
            n=25,
        )

        expected.append(
            len(masses) * integral / total_norm
        )

    expected = np.asarray(
        expected,
        dtype=float,
    )

    if (
        np.any(~np.isfinite(expected))
        or np.any(expected <= 0.0)
    ):
        return 1e100

    nll = np.sum(
        expected
        - observed * np.log(expected)
        + gammaln(observed + 1.0)
    )

    return (
        float(nll)
        if np.isfinite(nll)
        else 1e100
    )


def fit_model(masses, model_name, fit_cfg):
    """Fit ExpPoly2 or Bern3 and calculate B in 120--130."""
    if model_name == "ExpPoly2":
        initial = np.array([
            0.0,
            0.0,
            0.0,
        ])

        bounds = [
            (-20.0, 20.0),
            (-10.0, 10.0),
            (-10.0, 10.0),
        ]

    elif model_name == "Bern3":
        initial = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
        ])

        bounds = [
            (-10.0, 10.0),
            (-10.0, 10.0),
            (-10.0, 10.0),
            (-10.0, 10.0),
        ]

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    result = minimize(
        poisson_nll,
        x0=initial,
        args=(
            masses,
            model_name,
            fit_cfg,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 3000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    params = np.asarray(
        result.x,
        dtype=float,
    )

    total_norm = sideband_integral(
        params,
        model_name,
        fit_cfg,
    )

    func = lambda x: model_raw(
        x,
        model_name,
        params,
        fit_cfg,
    )

    sr_norm = numerical_integral(
        func,
        fit_cfg["signal_lo"],
        fit_cfg["signal_hi"],
        n=1000,
    )

    b_sr = (
        len(masses)
        * sr_norm
        / total_norm
        if total_norm > 0.0
        else 0.0
    )

    return {
        "name": model_name,
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "params": params,
        "sideband_norm": float(total_norm),
        "sr_norm": float(sr_norm),
        "b_sr": float(b_sr),
    }


# =============================================================================
# 7. Plot
# =============================================================================

def plot_fit(
    masses,
    fit_result,
    c2,
    fit_cfg,
):
    """Plot Data sideband and fitted function, including SR extrapolation."""
    model_name = fit_result["name"]

    bins = np.arange(
        fit_cfg["fit_min"],
        fit_cfg["fit_max"] + PLOT_BIN_WIDTH,
        PLOT_BIN_WIDTH,
    )

    counts, edges = np.histogram(
        masses,
        bins=bins,
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    errors = np.sqrt(counts)

    fig, ax = plt.subplots(
        figsize=(8.2, 5.5)
    )

    ax.errorbar(
        centers,
        counts,
        yerr=errors,
        fmt="o",
        ms=3.5,
        lw=1,
        capsize=2,
        label="Data sideband",
    )

    x = np.linspace(
        fit_cfg["fit_min"],
        fit_cfg["fit_max"],
        2000,
    )

    shape = model_raw(
        x,
        model_name,
        fit_result["params"],
        fit_cfg,
    )

    y = (
        len(masses)
        * shape
        / fit_result["sideband_norm"]
    )

    ax.plot(
        x,
        y * PLOT_BIN_WIDTH,
        linewidth=2.0,
        label=model_name,
    )

    ax.axvspan(
        fit_cfg["signal_lo"],
        fit_cfg["signal_hi"],
        alpha=0.12,
        label="120–130 GeV",
    )

    ax.set_xlabel(
        r"$m_{\ell\ell\gamma}$ [GeV]"
    )

    ax.set_ylabel(
        f"Data events / {PLOT_BIN_WIDTH:.0f} GeV"
    )

    ax.set_title(
        f"GNN high: score > {c2:.3f} — {model_name}"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    plt.tight_layout()

    out_path = (
        OUT_DIR
        / f"fit_high_{model_name}.png"
    )

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


# =============================================================================
# 8. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Method 1: fit Data sidebands with ExpPoly2 and Bern3 "
            "and extrapolate to 120--130 GeV."
        )
    )

    parser.add_argument(
        "--c1",
        type=float,
        default=DEFAULT_C1,
        help="GNN medium/high lower boundary, kept for bookkeeping.",
    )

    parser.add_argument(
        "--c2",
        type=float,
        default=DEFAULT_C2,
        help="GNN high threshold: score > c2.",
    )

    parser.add_argument(
        "--fit-min",
        type=float,
        default=DEFAULT_FIT_MIN,
        help="Left edge of the fitted sideband.",
    )

    parser.add_argument(
        "--fit-max",
        type=float,
        default=DEFAULT_FIT_MAX,
        help="Right edge of the fitted sideband.",
    )

    args = parser.parse_args()

    if args.c2 <= args.c1:
        raise ValueError(
            f"Require c2 > c1, got c1={args.c1}, c2={args.c2}"
        )

    fit_cfg = {
        "fit_min": float(args.fit_min),
        "signal_lo": float(DEFAULT_SIGNAL_LO),
        "signal_hi": float(DEFAULT_SIGNAL_HI),
        "fit_max": float(args.fit_max),
    }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 90)
    print("METHOD 1 — DATA SIDEBAND MASS FIT")
    print("=" * 90)

    print(
        f"GNN category: HIGH, score > {args.c2:.6f}"
    )

    print(
        "Fit sidebands: "
        f"{fit_cfg['fit_min']:.1f}–{fit_cfg['signal_lo']:.1f} GeV "
        "and "
        f"{fit_cfg['signal_hi']:.1f}–{fit_cfg['fit_max']:.1f} GeV"
    )

    print(
        "Signal region: "
        f"{fit_cfg['signal_lo']:.1f}–{fit_cfg['signal_hi']:.1f} GeV"
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    data = load_data()

    data_sb = select_data_sideband(
        data,
        args.c2,
        fit_cfg,
    )

    data_sr = select_data_signal_region(
        data,
        args.c2,
        fit_cfg,
    )

    print("\n" + "-" * 90)
    print("DATA")
    print("-" * 90)

    print(
        f"Data sideband N = {len(data_sb)}"
    )

    print(
        f"Data 120–130 N  = {len(data_sr)} "
        "(diagnostic only; NOT used as B)"
    )

    if len(data_sb) == 0:
        raise RuntimeError(
            "No Data sideband events in the selected GNN high category."
        )

    # -------------------------------------------------------------------------
    # Fit two simple functions
    # -------------------------------------------------------------------------

    fit_results = {}

    for model_name in [
        "ExpPoly2",
        "Bern3",
    ]:
        print("\n" + "=" * 90)
        print(f"FIT WITH {model_name}")
        print("=" * 90)

        fit_result = fit_model(
            data_sb,
            model_name,
            fit_cfg,
        )

        fit_results[model_name] = fit_result

        print(
            f"success = {fit_result['success']}"
        )

        print(
            f"message = {fit_result['message']}"
        )

        print(
            "parameters = "
            + np.array2string(
                fit_result["params"],
                precision=8,
            )
        )

        print(
            f"N sideband = {len(data_sb)}"
        )

        print(
            f"Predicted B(120–130) = "
            f"{fit_result['b_sr']:.8f}"
        )

        plot_fit(
            data_sb,
            fit_result,
            args.c2,
            fit_cfg,
        )

    # -------------------------------------------------------------------------
    # Signal
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("SIGNAL — TT H, 2-FOLD OOF")
    print("=" * 90)

    signal = load_tth_signal_oof(
        fit_cfg,
    )

    signal_high = (
        signal[SCORE] > args.c2
    )

    s, sigma_s, n_s = weighted_yield(
        signal[WEIGHT][signal_high]
    )

    print(
        f"ttH high raw N = {n_s}"
    )

    print(
        f"ttH high S     = {s:.8f}"
    )

    print(
        f"MC sigma(S)    = {sigma_s:.8f}"
    )

    # -------------------------------------------------------------------------
    # Old MC background comparison
    # -------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("OLD MC BACKGROUND — COMPARISON ONLY")
    print("=" * 90)

    old_mc_b = None
    old_mc_sigma = None

    try:
        background = load_background_mc_oof(
            fit_cfg,
        )

        if background is not None:
            high = (
                background[SCORE] > args.c2
            )

            old_mc_b, old_mc_sigma, old_mc_n = weighted_yield(
                background[WEIGHT][high]
            )

            print(
                f"Old MC high raw N = {old_mc_n}"
            )

            print(
                f"Old MC high B     = {old_mc_b:.8f}"
            )

            print(
                f"Old MC sigma(B)   = {old_mc_sigma:.8f}"
            )

            if old_mc_b != 0.0:
                print(
                    f"Old MC relative B uncertainty = "
                    f"{abs(old_mc_sigma / old_mc_b):.4f}"
                )

        else:
            print("No old background MC available.")

    except Exception as exc:
        print(
            f"WARNING: old MC comparison failed: {exc}",
            file=sys.stderr,
        )

    # -------------------------------------------------------------------------
    # Final Method 1 table
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("FINAL METHOD 1 RESULTS")
    print("=" * 100)

    header = (
        f"{'Method':<15}"
        f"{'S':>14}"
        f"{'B(120–130)':>18}"
        f"{'Z':>14}"
    )

    print(header)
    print("-" * len(header))

    summary_lines = []

    for model_name in [
        "ExpPoly2",
        "Bern3",
    ]:
        b = fit_results[model_name]["b_sr"]

        z = asimov_z(
            s,
            b,
        )

        print(
            f"{model_name:<15}"
            f"{s:>14.6f}"
            f"{b:>18.6f}"
            f"{z:>14.6f}"
        )

        summary_lines.append(
            f"{model_name}: "
            f"S={s:.8f}, "
            f"B_120_130={b:.8f}, "
            f"Z={z:.8f}"
        )

    print("-" * len(header))

    if old_mc_b is not None:
        old_z = asimov_z(
            s,
            old_mc_b,
        )

        print(
            f"{'Old MC':<15}"
            f"{s:>14.6f}"
            f"{old_mc_b:>18.6f}"
            f"{old_z:>14.6f}"
        )

        summary_lines.append(
            f"Old MC: "
            f"S={s:.8f}, "
            f"B_120_130={old_mc_b:.8f}, "
            f"Z={old_z:.8f}"
        )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    summary_path = (
        OUT_DIR
        / "fit_high_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Method 1: Data sideband mass fit\n"
        )
        f.write(
            "=================================\n\n"
        )

        f.write(
            f"GNN high threshold: score > {args.c2:.8f}\n"
        )

        f.write(
            f"Fit sidebands: "
            f"{fit_cfg['fit_min']:.1f}-{fit_cfg['signal_lo']:.1f} "
            f"and "
            f"{fit_cfg['signal_hi']:.1f}-{fit_cfg['fit_max']:.1f} GeV\n"
        )

        f.write(
            f"Signal region: "
            f"{fit_cfg['signal_lo']:.1f}-{fit_cfg['signal_hi']:.1f} GeV\n\n"
        )

        f.write(
            f"N_Data_sideband = {len(data_sb)}\n"
        )

        f.write(
            f"N_Data_120_130 = {len(data_sr)} "
            "(diagnostic only)\n\n"
        )

        f.write(
            f"ttH OOF high raw N = {n_s}\n"
        )

        f.write(
            f"ttH OOF high S = {s:.8f}\n"
        )

        f.write(
            f"ttH MC sigma(S) = {sigma_s:.8f}\n\n"
        )

        for line in summary_lines:
            f.write(line + "\n")

    print(
        f"\nSaved summary: {summary_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
