#!/usr/bin/env python3
"""
Clean 2-fold OOF significance study for ttH vs other.

OOF convention
--------------
- fold0 model predicts parity = 0 events
- fold1 model predicts parity = 1 events

This script:
1. Loads fold0/fold1 scored ROOT files.
2. Builds true 2-fold OOF MC samples.
3. Computes cut-based significance.
4. Optimizes GNN category boundaries with Optuna and grid scan.
5. Prints high-category background composition.
6. Produces BOTH weighted and avg-weighted m_llg plots for:
   - GNN high category
   - original cut-based ttHlep category

Data sideband is read from the fold0 scored Data directory and is used only
as a category-stability constraint. It is NOT split into OOF folds.
"""

import sys
from pathlib import Path

import numpy as np
import optuna
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================

OUT_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results_lypt"
)
# DATA_DIR_FOLD0 = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored_ttH_vs_other_run3v2_fold0_0828")
DATA_DIR_FOLD0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
)
# DATA_DIR_FOLD0 = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_full_decorr_scored_fold0")

# DATA_DIR_FOLD1 = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored_ttH_vs_other_run3v2_fold1_0828")

DATA_DIR_FOLD1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold1"
)
# DATA_DIR_FOLD1 = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_full_decorr_scored_fold1")

# Data sideband: keep the original convention and use fold0 Data directly.
DATA_DIR = DATA_DIR_FOLD0 / "Data"
# DATA_DIR = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_full_decorr_scored_data/Data")
TREE = "merged_SR_Sideband"

WEIGHT = "weight_lumi"
MASS = "m_llg"
SCORE = "ttH_gnn_score"
IS_SR = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR = "photon_eta_fold"

WEIGHT_AVG = "weight_lumi_avg"
BRANCHES = [
    WEIGHT,
    WEIGHT_AVG,
    MASS,
    SCORE,
    IS_SR,
    TREE_TAG,
    FOLDVAR,
]

DATA_BRANCHES = [
    WEIGHT,
    SCORE,
    IS_SR,
    TREE_TAG,
]

# True OOF convention:
# fold0 config: test=[0], train=[1] -> evaluate parity 0
# fold1 config: test=[1], train=[0] -> evaluate parity 1
FOLD0_OOF_PARITY = 0
FOLD1_OOF_PARITY = 1

MASS_LO = 120.0
MASS_HI = 130.0

SR_TAG_MAP = {
    0: "ttHhad",
    2: "ttHlep",
    4: "untag",
    6: "VH3l",
    8: "VHpTmiss",
}

VALID_SR_TAGS = np.array(list(SR_TAG_MAP.keys()))
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


# ---- GNN category optimization ----

C1_MIN = 0.10
C1_MAX = 0.95

C2_MIN = 0.20
C2_MAX = 0.999

MIN_GAP = 0.001

N_TRIALS = 1000
N_STARTUP_TRIALS = 50
RANDOM_STATE = 12345

C1_VALUES = np.arange(0.10, 0.951, 0.01)
# np.arange's stop is exclusive; append C2_MAX so the grid endpoint
# aligns with the optimization ceiling (0.999, which is off the 0.005 grid).
C2_VALUES = np.append(np.arange(C2_MIN, C2_MAX, 0.005), C2_MAX)

# Optional stability requirements.
# Set to zero to disable them.
MIN_B_NEFF = 0.0
MIN_HIGH_SR_TOTAL_YIELD = 0.0

# Require a non-zero weighted Data sideband yield in the GNN high category.
# Set to 0.0 to require strictly > 0.
MIN_HIGH_DATA_SB_YIELD = 0


# =============================================================================
# 2. Statistics
# =============================================================================

def asimov_z(s, b):
    """Asimov significance."""
    if s <= 0.0 or b <= 0.0:
        return 0.0

    x = 2.0 * ((s + b) * np.log1p(s / b) - s)
    return float(np.sqrt(x)) if x > 0.0 else 0.0


def yield_info(weights):
    """
    Weighted yield information.

    Absolute MC statistical uncertainty:
        sigma = sqrt(sum w^2)

    Effective statistics:
        Neff = (sum w)^2 / sum(w^2)
    """
    w = np.asarray(weights, dtype=float)

    if w.size == 0:
        return {
            "sum": 0.0,
            "err": 0.0,
            "n": 0,
            "nneg": 0,
            "neff": 0.0,
        }

    sumw = float(np.sum(w))
    sumw2 = float(np.sum(w * w))

    neff = (sumw * sumw / sumw2) if sumw2 > 0.0 else 0.0

    return {
        "sum": sumw,
        "err": float(np.sqrt(sumw2)),
        "n": int(w.size),
        "nneg": int(np.sum(w < 0.0)),
        "neff": float(neff),
    }


def z_uncertainty(s, b, sigma_s, sigma_b):
    """Linear error propagation for Asimov Z."""
    z = asimov_z(s, b)

    if z <= 0.0:
        return 0.0, 0.0, 0.0

    log_term = np.log1p(s / b)

    dz_ds = log_term / z
    dz_db = (log_term - s / b) / z

    sigma_z_s = abs(dz_ds * sigma_s)
    sigma_z_b = abs(dz_db * sigma_b)
    sigma_z = np.hypot(sigma_z_s, sigma_z_b)

    return float(sigma_z), float(sigma_z_s), float(sigma_z_b)


def combined_z(results):
    """Quadrature combination of independent category significances."""
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z = float(np.sqrt(z2))

    if z <= 0.0:
        return 0.0, 0.0

    sigma2 = sum(
        ((r["Z"] / z) * r["sigma_Z"]) ** 2
        for r in results.values()
    )

    return z, float(np.sqrt(sigma2))


# =============================================================================
# 3. ROOT helpers
# =============================================================================

def empty_arrays():
    return {
        branch: np.array([])
        for branch in BRANCHES
    }


def concatenate(arrays):
    out = {}

    for branch in BRANCHES:
        pieces = [
            arr[branch]
            for arr in arrays
            if arr[branch].size > 0
        ]

        out[branch] = (
            np.concatenate(pieces)
            if pieces
            else np.array([])
        )

    return out


def apply_mask(arr, mask):
    return {
        branch: arr[branch][mask]
        for branch in BRANCHES
    }


def load_file(path):
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return empty_arrays()

            tree = f[TREE]

            missing = [
                branch
                for branch in BRANCHES
                if branch not in tree.keys()
            ]

            if missing:
                print(
                    f"WARNING: {path} missing branches: {missing}",
                    file=sys.stderr,
                )
                return empty_arrays()

            return tree.arrays(BRANCHES, library="np")

    except Exception as exc:
        print(
            f"WARNING: cannot read {path}: {exc}",
            file=sys.stderr,
        )
        return empty_arrays()


def load_group(data_dir, subdir, years):
    pieces = []

    print(f"Loading {subdir} from {data_dir.name} ...")

    for year in years:
        path = data_dir / subdir / f"{year}.root"

        if not path.exists():
            print(f"  missing: {path}")
            continue

        arr = load_file(path)

        if arr[WEIGHT].size > 0:
            pieces.append(arr)

    return concatenate(pieces)


def load_all(data_dir):
    signal = load_group(
        data_dir,
        SIGNAL_GROUP[0],
        SIGNAL_GROUP[1],
    )

    # Keep per-group parts so downstream code (e.g. the high-background
    # breakdown) can reuse them without re-reading the files from disk.
    background_parts = [
        (subdir, load_group(data_dir, subdir, years))
        for subdir, years in BACKGROUND_GROUPS
    ]

    background = concatenate(
        [
            arr
            for _, arr in background_parts
            if arr[WEIGHT].size > 0
        ]
    )

    return signal, background, background_parts


def load_data_sideband(data_dir):
    """
    Load Data sideband from fold0/Data.

    Data is NOT OOF-split here. This matches the original workflow:
    use Data only to require a non-zero sideband yield in the GNN category.
    """
    files = sorted(data_dir.rglob("*.root"))

    if not files:
        raise RuntimeError(
            f"No ROOT files found under Data directory: {data_dir}"
        )

    weights = []
    scores = []
    tags = []
    n_files_used = 0

    print(f"\nLoading Data sideband from {data_dir} ...")

    for path in files:
        try:
            with uproot.open(str(path)) as f:
                if TREE not in f:
                    continue

                tree = f[TREE]

                missing = [
                    branch
                    for branch in DATA_BRANCHES
                    if branch not in tree.keys()
                ]

                if missing:
                    print(
                        f"WARNING: {path} missing Data branches: {missing}",
                        file=sys.stderr,
                    )
                    continue

                arr = tree.arrays(
                    DATA_BRANCHES,
                    library="np",
                )

        except Exception as exc:
            print(
                f"WARNING: cannot read Data file {path}: {exc}",
                file=sys.stderr,
            )
            continue

        mask = (
            (arr[IS_SR] == 0)
            & np.isfinite(arr[WEIGHT])
            & np.isfinite(arr[SCORE])
        )

        if np.any(mask):
            weights.append(
                np.asarray(arr[WEIGHT][mask], dtype=float)
            )
            scores.append(
                np.asarray(arr[SCORE][mask], dtype=float)
            )
            tags.append(
                np.asarray(arr[TREE_TAG][mask])
            )

        n_files_used += 1

    if not weights:
        raise RuntimeError(
            "No usable Data sideband events found with is_SR == 0."
        )

    weights = np.concatenate(weights)
    scores = np.concatenate(scores)
    tags = np.concatenate(tags)

    print(
        f"Data sideband: files used={n_files_used}, "
        f"N={weights.size}, "
        f"weighted yield={np.sum(weights):.6f}"
    )

    return weights, scores, tags


# =============================================================================
# 4. Event selections
# =============================================================================

def sr_physics_mask(arr):
    """
    SR events used for significance evaluation.
    """
    return (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
    )


def oof_sr_mask(arr, parity):
    """
    True OOF SR selection for a given scored directory.
    """
    return (
        sr_physics_mask(arr)
        & ((arr[FOLDVAR] % 2) == parity)
    )


def oof_fullrange_mask(arr, parity):
    """
    OOF selection over SR + sideband tree tags.
    Used only for mass-shape plots.
    """
    return (
        np.isin(arr[TREE_TAG], VALID_FULL_TAGS)
        & ((arr[FOLDVAR] % 2) == parity)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[MASS])
        & np.isfinite(arr[SCORE])
    )


def build_oof_sample(arr_fold0, arr_fold1, mask_builder):
    """
    Build true OOF sample:
      fold0 scored sample -> parity 0
      fold1 scored sample -> parity 1
    """
    part0 = apply_mask(
        arr_fold0,
        mask_builder(arr_fold0, FOLD0_OOF_PARITY),
    )

    part1 = apply_mask(
        arr_fold1,
        mask_builder(arr_fold1, FOLD1_OOF_PARITY),
    )

    return concatenate([part0, part1])


# =============================================================================
# 5. Cut-based significance
# =============================================================================

def calculate_cutbased(signal, background, data_tags):
    results = {}

    for tag, name in SR_TAG_MAP.items():

        signal_mask = signal[TREE_TAG] == tag
        background_mask = background[TREE_TAG] == tag

        S = yield_info(signal[WEIGHT][signal_mask])
        B = yield_info(background[WEIGHT][background_mask])

        # Sideband tag corresponding to this SR category is tag + 1.
        data_sb_n = int(np.sum(data_tags == tag + 1))

        z = asimov_z(S["sum"], B["sum"])

        sigma_z, sigma_z_s, sigma_z_b = z_uncertainty(
            S["sum"],
            B["sum"],
            S["err"],
            B["err"],
        )

        results[name] = {
            "S": S,
            "B": B,
            "DataSB": data_sb_n,
            "Z": z,
            "sigma_Z": sigma_z,
            "sigma_Z_from_S": sigma_z_s,
            "sigma_Z_from_B": sigma_z_b,
        }

    return results


def print_cutbased_table(results):
    z, sigma_z = combined_z(results)

    print("\n" + "=" * 125)
    print("CUT-BASED OOF")
    print("=" * 125)

    header = (
        f"{'Cat':<10}"
        f"{'S':>10}{'σS':>10}"
        f"{'B':>12}{'σB':>10}"
        f"{'NeffB':>10}"
        f"{'NB':>8}"
        f"{'Nneg':>8}"
        f"{'DataSB':>9}"
        f"{'Z':>10}"
        f"{'σZ(S)':>10}"
        f"{'σZ(B)':>10}"
        f"{'σZ':>10}"
    )

    print(header)
    print("-" * len(header))

    for name in SR_TAG_MAP.values():
        r = results[name]
        S = r["S"]
        B = r["B"]

        print(
            f"{name:<10}"
            f"{S['sum']:>10.4f}{S['err']:>10.4f}"
            f"{B['sum']:>12.4f}{B['err']:>10.4f}"
            f"{B['neff']:>10.2f}"
            f"{B['n']:>8d}"
            f"{B['nneg']:>8d}"
            f"{r['DataSB']:>9d}"
            f"{r['Z']:>10.4f}"
            f"{r['sigma_Z_from_S']:>10.4f}"
            f"{r['sigma_Z_from_B']:>10.4f}"
            f"{r['sigma_Z']:>10.4f}"
        )

    print("-" * len(header))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}")


# =============================================================================
# 6. GNN 3-category evaluation
# =============================================================================

def evaluate_3cat(signal, background, data_weights, data_scores, c1, c2):
    signal_score = signal[SCORE]
    background_score = background[SCORE]

    category_masks = {
        "high": (
            signal_score > c2,
            background_score > c2,
            data_scores > c2,
        ),
        "medium": (
            (signal_score > c1) & (signal_score <= c2),
            (background_score > c1) & (background_score <= c2),
            (data_scores > c1) & (data_scores <= c2),
        ),
        "low": (
            signal_score <= c1,
            background_score <= c1,
            data_scores <= c1,
        ),
    }

    results = {}

    for category, (signal_mask, background_mask, data_mask) in category_masks.items():

        S = yield_info(signal[WEIGHT][signal_mask])
        B = yield_info(background[WEIGHT][background_mask])

        data_sb_yield = float(
            np.sum(data_weights[data_mask])
        )

        # Invalid if weighted background yield is non-positive.
        if B["sum"] <= 0.0:
            return {}, 0.0, False

        # Optional MC-stat stability requirement.
        if B["neff"] < MIN_B_NEFF:
            return {}, 0.0, False

        # Optional requirement on high-category total SR yield.
        if category == "high":
            if S["sum"] + B["sum"] < MIN_HIGH_SR_TOTAL_YIELD:
                return {}, 0.0, False

            if data_sb_yield <= MIN_HIGH_DATA_SB_YIELD:
                return {}, 0.0, False

        z = asimov_z(S["sum"], B["sum"])

        sigma_z, sigma_z_s, sigma_z_b = z_uncertainty(
            S["sum"],
            B["sum"],
            S["err"],
            B["err"],
        )

        results[category] = {
            "S": S,
            "B": B,
            "DataSBYield": data_sb_yield,
            "Z": z,
            "sigma_Z": sigma_z,
            "sigma_Z_from_S": sigma_z_s,
            "sigma_Z_from_B": sigma_z_b,
        }

    z_total, _ = combined_z(results)

    return results, z_total, True


def print_gnn_results(title, results, c1, c2):
    z, sigma_z = combined_z(results)

    print("\n" + "=" * 116)
    print(f"{title}: c1={c1:.6f}, c2={c2:.6f}")
    print("=" * 116)

    header = (
        f"{'Cat':<8}"
        f"{'S':>10}{'σS':>10}"
        f"{'B':>12}{'σB':>10}"
        f"{'NeffB':>10}"
        f"{'NB':>8}"
        f"{'Nneg':>8}"
        f"{'DataSBYield':>14}"
        f"{'Z':>10}"
        f"{'σZ(S)':>10}"
        f"{'σZ(B)':>10}"
        f"{'σZ':>10}"
    )

    print(header)
    print("-" * len(header))

    for name in ["high", "medium", "low"]:
        r = results[name]
        S = r["S"]
        B = r["B"]

        print(
            f"{name:<8}"
            f"{S['sum']:>10.4f}{S['err']:>10.4f}"
            f"{B['sum']:>12.4f}{B['err']:>10.4f}"
            f"{B['neff']:>10.2f}"
            f"{B['n']:>8d}"
            f"{B['nneg']:>8d}"
            f"{r['DataSBYield']:>14.4f}"
            f"{r['Z']:>10.4f}"
            f"{r['sigma_Z_from_S']:>10.4f}"
            f"{r['sigma_Z_from_B']:>10.4f}"
            f"{r['sigma_Z']:>10.4f}"
        )

    print("-" * len(header))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}")


# =============================================================================
# 7. Optuna optimization
# =============================================================================

def decode_trial(trial):
    c1 = trial.suggest_float(
        "c1",
        C1_MIN,
        C1_MAX,
    )

    c2_low = max(
        C2_MIN,
        c1 + MIN_GAP,
    )

    if c2_low >= C2_MAX:
        return None, None

    c2 = trial.suggest_float(
        "c2",
        c2_low,
        C2_MAX,
    )

    return float(c1), float(c2)


def optimize_bayesian(signal, background, data_weights, data_scores):
    def objective(trial):
        c1, c2 = decode_trial(trial)

        if c1 is None:
            return -1.0

        results, z, valid = evaluate_3cat(
            signal,
            background,
            data_weights,
            data_scores,
            c1,
            c2,
        )

        trial.set_user_attr("valid", valid)

        return z if valid else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=N_STARTUP_TRIALS,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    print(
        f"\nRunning Optuna: "
        f"{N_TRIALS} trials"
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
    )

    valid_trials = [
        trial
        for trial in study.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.user_attrs.get("valid", False)
        )
    ]

    if not valid_trials:
        raise RuntimeError(
            "No valid Optuna trial found."
        )

    best = max(
        valid_trials,
        key=lambda trial: trial.value,
    )

    c1 = float(best.params["c1"])
    c2 = float(best.params["c2"])

    results, _, _ = evaluate_3cat(
        signal,
        background,
        data_weights,
        data_scores,
        c1,
        c2,
    )

    return c1, c2, results


# =============================================================================
# 8. Grid scan
# =============================================================================

def scan_gnn(signal, background, data_weights, data_scores):
    best = None
    best_z = -1.0

    tested = 0
    valid_points = 0

    for c1 in C1_VALUES:
        for c2 in C2_VALUES:

            if c2 <= c1 + MIN_GAP:
                continue

            tested += 1

            results, z, valid = evaluate_3cat(
                signal,
                background,
                data_weights,
                data_scores,
                c1,
                c2,
            )

            if not valid:
                continue

            valid_points += 1

            if z > best_z:
                best_z = z
                best = (
                    results,
                    float(c1),
                    float(c2),
                )

    print(
        f"\nGrid scan: "
        f"tested={tested}, valid={valid_points}"
    )

    if best is None:
        return None, 0.0, None, None

    results, c1, c2 = best

    return results, best_z, c1, c2


# =============================================================================
# 9. High-category background breakdown
# =============================================================================

def print_high_background_breakdown(
    background_parts_fold0,
    background_parts_fold1,
    c2,
):
    print("\n" + "=" * 92)
    print(
        f"HIGH BACKGROUND BREAKDOWN "
        f"(OOF, score > {c2:.6f})"
    )
    print("=" * 92)

    header = (
        f"{'Sample':<30}"
        f"{'B':>12}"
        f"{'σB':>12}"
        f"{'Neff':>10}"
        f"{'N':>8}"
        f"{'Nneg':>8}"
    )

    print(header)
    print("-" * len(header))

    # Reuse the already-loaded per-group samples instead of re-reading the
    # files from disk. Both folds share the same BACKGROUP_GROUPS ordering,
    # so we zip them together; the subdir label comes from fold0's parts.
    for (subdir, arr_fold0), (_, arr_fold1) in zip(
        background_parts_fold0,
        background_parts_fold1,
    ):

        oof = build_oof_sample(
            arr_fold0,
            arr_fold1,
            oof_sr_mask,
        )

        high_mask = oof[SCORE] > c2

        info = yield_info(
            oof[WEIGHT][high_mask]
        )

        print(
            f"{subdir:<30}"
            f"{info['sum']:>12.4f}"
            f"{info['err']:>12.4f}"
            f"{info['neff']:>10.2f}"
            f"{info['n']:>8d}"
            f"{info['nneg']:>8d}"
        )


def print_high_background_avgweight_diagnostics(
    background_parts_fold0,
    background_parts_fold1,
    c2,
):
    print("\n" + "=" * 112)
    print(
        f"GNN HIGH BACKGROUND AVG-WEIGHT DIAGNOSTICS "
        f"(OOF full range, score > {c2:.6f})"
    )
    print("=" * 112)

    sample_rows = []
    total_mass = []
    total_weights = []

    for (subdir, arr_fold0), (_, arr_fold1) in zip(
        background_parts_fold0,
        background_parts_fold1,
    ):
        oof = build_oof_sample(
            arr_fold0,
            arr_fold1,
            oof_fullrange_mask,
        )

        high_mask = oof[SCORE] > c2
        weights = np.asarray(oof[WEIGHT_AVG][high_mask], dtype=float)
        masses = np.asarray(oof[MASS][high_mask], dtype=float)
        info = yield_info(weights)

        sample_rows.append((subdir, info))

        if weights.size > 0:
            total_mass.append(masses)
            total_weights.append(weights)

    header = (
        f"{'Sample':<30}"
        f"{'RawN':>8}"
        f"{'sum(w_avg)':>14}"
        f"{'sqrt(sumw2)':>14}"
        f"{'Neff':>10}"
        f"{'Neff/N':>10}"
        f"{'avg w':>12}"
    )

    print(header)
    print("-" * len(header))

    for subdir, info in sample_rows:
        avg_weight = info["sum"] / info["n"] if info["n"] > 0 else 0.0
        neff_over_n = info["neff"] / info["n"] if info["n"] > 0 else 0.0

        print(
            f"{subdir:<30}"
            f"{info['n']:>8d}"
            f"{info['sum']:>14.6f}"
            f"{info['err']:>14.6f}"
            f"{info['neff']:>10.2f}"
            f"{neff_over_n:>10.3f}"
            f"{avg_weight:>12.6g}"
        )

    if not total_weights:
        print("No high-category background events found for avg-weight diagnostics.")
        return

    total_mass = np.concatenate(total_mass)
    total_weights = np.concatenate(total_weights)

    total_info = yield_info(total_weights)
    avg_weight = total_info["sum"] / total_info["n"] if total_info["n"] > 0 else 0.0
    neff_over_n = total_info["neff"] / total_info["n"] if total_info["n"] > 0 else 0.0

    print("-" * len(header))
    print(
        f"{'TOTAL':<30}"
        f"{total_info['n']:>8d}"
        f"{total_info['sum']:>14.6f}"
        f"{total_info['err']:>14.6f}"
        f"{total_info['neff']:>10.2f}"
        f"{neff_over_n:>10.3f}"
        f"{avg_weight:>12.6g}"
    )

    bin_width = 4.0
    bins = np.arange(100.0, 180.0 + bin_width, bin_width)

    print("\n" + "-" * 88)
    print("GNN HIGH BACKGROUND AVG-WEIGHT BY 4 GEV BIN")
    print("-" * 88)

    bin_header = (
        f"{'m_llg bin':<15}"
        f"{'RawN':>8}"
        f"{'sum(w_avg)':>14}"
        f"{'sqrt(sumw2)':>14}"
        f"{'rel err':>10}"
        f"{'Neff':>10}"
    )

    print(bin_header)
    print("-" * len(bin_header))

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (total_mass >= lo) & (total_mass < hi)
        weights = total_weights[mask]

        if weights.size == 0:
            continue

        info = yield_info(weights)
        rel_err = info["err"] / info["sum"] if info["sum"] > 0.0 else 0.0

        print(
            f"{lo:5.1f}-{hi:<5.1f}"
            f"{info['n']:>8d}"
            f"{info['sum']:>14.6f}"
            f"{info['err']:>14.6f}"
            f"{rel_err:>10.3f}"
            f"{info['neff']:>10.2f}"
        )


# =============================================================================
# 10. Histogram helpers
# =============================================================================

def weighted_histogram(x, weights, bins):
    """
    Weighted histogram:
      content = sum(w)
      error   = sqrt(sum(w^2))
    """
    y, edges = np.histogram(
        x,
        bins=bins,
        weights=weights,
    )

    sumw2, _ = np.histogram(
        x,
        bins=bins,
        weights=weights * weights,
    )

    err = np.sqrt(sumw2)

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    return y, err, edges, centers


def make_mass_plot_avg_weight(
    signal_mass,
    background_mass,
    signal_weights_avg,
    background_weights_avg,
    out_path,
    title,
):
    """
    Mass plot using weight_lumi_avg.

    This is NOT raw event count.
    Each event is filled with weight_lumi_avg.
    """

    BIN_WIDTH = 4.0

    bins = np.arange(
        100.0,
        180.0 + BIN_WIDTH,
        BIN_WIDTH,
    )

    total_mass = np.concatenate([
        signal_mass,
        background_mass,
    ])

    total_weights = np.concatenate([
        signal_weights_avg,
        background_weights_avg,
    ])

    total_y, total_err, _, centers = weighted_histogram(
        total_mass,
        total_weights,
        bins,
    )

    fig, ax = plt.subplots(
        figsize=(7.8, 5.4)
    )

    # Background
    ax.hist(
        background_mass,
        bins=bins,
        weights=background_weights_avg,
        histtype="stepfilled",
        alpha=0.35,
        label="Background",
    )

    # ttH signal
    ax.hist(
        signal_mass,
        bins=bins,
        weights=signal_weights_avg,
        histtype="step",
        linewidth=1.8,
        label="ttH signal",
    )

    # S+B
    ax.hist(
        total_mass,
        bins=bins,
        weights=total_weights,
        histtype="step",
        linewidth=2.2,
        label="S + B",
    )

    # uncertainty
    ax.errorbar(
        centers,
        total_y,
        yerr=total_err,
        fmt="o",
        ms=3,
        lw=1,
        capsize=2,
    )

    ax.axvspan(
        MASS_LO,
        MASS_HI,
        alpha=0.10,
        label="120–130 GeV",
    )

    ax.set_xlabel(
        r"$m_{\ell\ell\gamma}$ [GeV]"
    )

    ax.set_ylabel(
        f"Avg-weighted events / {BIN_WIDTH:.0f} GeV"
    )

    ax.set_title(title)

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    plt.tight_layout()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")

def make_mass_plot_weighted(
    signal_mass,
    background_mass,
    signal_weights,
    background_weights,
    out_path,
    title,
):
    """
    Weighted m_llg plot.

    Each event contributes weight_lumi.

    Histogram content:
        sum(w)

    Statistical uncertainty:
        sqrt(sum(w^2))
    """

    BIN_WIDTH = 4.0

    bins = np.arange(
        100.0,
        180.0 + BIN_WIDTH,
        BIN_WIDTH,
    )

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------

    signal_y, signal_err, _, centers = weighted_histogram(
        signal_mass,
        signal_weights,
        bins,
    )

    background_y, background_err, _, _ = weighted_histogram(
        background_mass,
        background_weights,
        bins,
    )

    total_mass = np.concatenate([
        signal_mass,
        background_mass,
    ])

    total_weights = np.concatenate([
        signal_weights,
        background_weights,
    ])

    total_y, total_err, _, _ = weighted_histogram(
        total_mass,
        total_weights,
        bins,
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.8, 5.4)
    )

    # Background
    ax.hist(
        background_mass,
        bins=bins,
        weights=background_weights,
        histtype="stepfilled",
        alpha=0.35,
        label="Background",
    )

    # ttH signal
    ax.hist(
        signal_mass,
        bins=bins,
        weights=signal_weights,
        histtype="step",
        linewidth=1.8,
        label="ttH signal",
    )

    # S+B
    ax.hist(
        total_mass,
        bins=bins,
        weights=total_weights,
        histtype="step",
        linewidth=2.2,
        label="S + B",
    )

    # Statistical uncertainty on S+B
    ax.errorbar(
        centers,
        total_y,
        yerr=total_err,
        fmt="o",
        ms=3,
        lw=1,
        capsize=2,
    )

    ax.axvspan(
        MASS_LO,
        MASS_HI,
        alpha=0.10,
        label="120–130 GeV",
    )

    ax.set_xlabel(
        r"$m_{\ell\ell\gamma}$ [GeV]"
    )

    ax.set_ylabel(
        f"Weighted events / {BIN_WIDTH:.0f} GeV"
    )

    ax.set_title(title)

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    plt.tight_layout()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


# =============================================================================
# 11. GNN-high mass plots
# =============================================================================

def plot_gnn_high_mass(
    signal_fold0,
    background_fold0,
    signal_fold1,
    background_fold1,
    c1,
):
    """
    Plot m_llg distribution for the GNN-high category.

    Two independent plots are produced:

    1. Unweighted:
       every event counts as 1.

    2. Weighted:
       every event is filled with weight_lumi.
    """

    # Build full-range OOF sample
    signal = build_oof_sample(
        signal_fold0,
        signal_fold1,
        oof_fullrange_mask,
    )

    background = build_oof_sample(
        background_fold0,
        background_fold1,
        oof_fullrange_mask,
    )

    # GNN high selection
    signal_high = (
        signal[SCORE] > c1
    )

    background_high = (
        background[SCORE] > c1
    )

    # Selected masses
    signal_mass = signal[MASS][signal_high]

    background_mass = background[MASS][background_high]

    # Selected weights
    signal_weights = signal[WEIGHT][signal_high]

    background_weights = background[WEIGHT][background_high]

    signal_weights_avg = signal[WEIGHT_AVG][signal_high]
    background_weights_avg = background[WEIGHT_AVG][background_high]
    # -------------------------------------------------------------------------
    # SR diagnostics
    #
    # Use the same SR definition as sr_physics_mask (IS_SR == 1 + mass window
    # + VALID_SR_TAGS + finite weight/score), so the printed yields match the
    # significance evaluation. The full-range sample also contains sideband
    # tags, which must NOT be counted as SR even if they fall in 120-130 GeV.
    # -------------------------------------------------------------------------

    signal_sr = (
        (signal[IS_SR] == 1)
        & (signal[MASS] > MASS_LO)
        & (signal[MASS] < MASS_HI)
        & np.isin(signal[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(signal[WEIGHT])
        & np.isfinite(signal[SCORE])
        & signal_high
    )

    background_sr = (
        (background[IS_SR] == 1)
        & (background[MASS] > MASS_LO)
        & (background[MASS] < MASS_HI)
        & np.isin(background[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(background[WEIGHT])
        & np.isfinite(background[SCORE])
        & background_high
    )

    print("\n" + "=" * 70)

    print(
        f"GNN HIGH CATEGORY: "
        f"score > {c1:.6f}"
    )

    print("=" * 70)

    print(
        f"Weighted SR signal     = "
        f"{np.sum(signal[WEIGHT][signal_sr]):.6f}"
    )

    print(
        f"Weighted SR background = "
        f"{np.sum(background[WEIGHT][background_sr]):.6f}"
    )

    print(
        f"Unweighted SR signal N = "
        f"{np.sum(signal_sr)}"
    )

    print(
        f"Unweighted SR bkg N    = "
        f"{np.sum(background_sr)}"
    )

    print(
        f"Full-range signal N    = "
        f"{len(signal_mass)}"
    )

    print(
        f"Full-range bkg N       = "
        f"{len(background_mass)}"
    )

    # -------------------------------------------------------------------------
    # Avg-weighted plot
    # -------------------------------------------------------------------------

    make_mass_plot_avg_weight(
        signal_mass,
        background_mass,
        signal_weights_avg,
        background_weights_avg,
        OUT_DIR / "mllg_gnn_high_avgweight.png",
        title=(
            r"$m_{\ell\ell\gamma}$ shape — "
            r"GNN high ($w_{\mathrm{lumi,avg}}$)"
        ),
    )
    # -------------------------------------------------------------------------
    # Weighted plot
    # -------------------------------------------------------------------------

    make_mass_plot_weighted(
        signal_mass,
        background_mass,
        signal_weights,
        background_weights,
        OUT_DIR / "mllg_gnn_high_weighted.png",
        title=(
            r"$m_{\ell\ell\gamma}$ shape — "
            r"GNN high (weighted)"
        ),
    )


# =============================================================================
# 12. ttHlep mass plots
# =============================================================================

def tthlep_fullrange_mask(arr, parity):
    """
    Original cut-based ttHlep category.

    SR:
        source_tree_tag = 2

    Sideband:
        source_tree_tag = 3

    OOF:
        select only the parity corresponding to the model
        that did NOT train on this event.
    """

    return (
        np.isin(
            arr[TREE_TAG],
            np.array([2, 3]),
        )
        & ((arr[FOLDVAR] % 2) == parity)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[MASS])
    )


def plot_tthlep_mass(
    signal_fold0,
    background_fold0,
    signal_fold1,
    background_fold1,
):
    """
    Plot m_llg distribution for the original cut-based ttHlep category.

    Two independent plots are produced:

    1. Unweighted:
       histogram content = raw number of MC entries.

    2. Weighted:
       histogram content = sum(weight_lumi).
    """

    # -------------------------------------------------------------------------
    # Build OOF ttHlep SR + sideband sample
    # -------------------------------------------------------------------------

    signal = build_oof_sample(
        signal_fold0,
        signal_fold1,
        tthlep_fullrange_mask,
    )

    background = build_oof_sample(
        background_fold0,
        background_fold1,
        tthlep_fullrange_mask,
    )

    # -------------------------------------------------------------------------
    # Mass and weight arrays
    # -------------------------------------------------------------------------

    signal_mass = signal[MASS]

    background_mass = background[MASS]

    signal_weights = signal[WEIGHT]

    background_weights = background[WEIGHT]

    signal_weights_avg = signal[WEIGHT_AVG]
    background_weights_avg = background[WEIGHT_AVG]
    # -------------------------------------------------------------------------
    # Explicit SR selection for diagnostics
    #
    # Use the same SR definition as sr_physics_mask (IS_SR == 1 + mass window
    # + VALID_SR_TAGS + finite weight/score). For ttHlep the SR tag is 2.
    # Do not count sideband tag == 3 here.
    # -------------------------------------------------------------------------

    signal_sr = (
        (signal[IS_SR] == 1)
        & (signal[MASS] > MASS_LO)
        & (signal[MASS] < MASS_HI)
        & (signal[TREE_TAG] == 2)
        & np.isfinite(signal[WEIGHT])
        & np.isfinite(signal[SCORE])
    )

    background_sr = (
        (background[IS_SR] == 1)
        & (background[MASS] > MASS_LO)
        & (background[MASS] < MASS_HI)
        & (background[TREE_TAG] == 2)
        & np.isfinite(background[WEIGHT])
        & np.isfinite(background[SCORE])
    )

    print("\n" + "=" * 70)

    print(
        "CUT-BASED ttHlep CATEGORY"
    )

    print("=" * 70)

    # Weighted physical yields
    print(
        f"Weighted SR signal     = "
        f"{np.sum(signal[WEIGHT][signal_sr]):.6f}"
    )

    print(
        f"Weighted SR background = "
        f"{np.sum(background[WEIGHT][background_sr]):.6f}"
    )

    # Raw MC counts
    print(
        f"Unweighted SR signal N = "
        f"{np.sum(signal_sr)}"
    )

    print(
        f"Unweighted SR bkg N    = "
        f"{np.sum(background_sr)}"
    )

    # Full range raw MC counts
    print(
        f"Full-range signal N    = "
        f"{len(signal_mass)}"
    )

    print(
        f"Full-range bkg N       = "
        f"{len(background_mass)}"
    )

    # Full range weighted yields
    print(
        f"Full-range signal sumw = "
        f"{np.sum(signal_weights):.6f}"
    )

    print(
        f"Full-range bkg sumw    = "
        f"{np.sum(background_weights):.6f}"
    )

    # -------------------------------------------------------------------------
    # Avg-weighted
    # -------------------------------------------------------------------------

    make_mass_plot_avg_weight(
        signal_mass,
        background_mass,
        signal_weights_avg,
        background_weights_avg,
        OUT_DIR / "mllg_tthlep_avgweight.png",
        title=(
            r"$m_{\ell\ell\gamma}$ shape — "
            r"ttHlep ($w_{\mathrm{lumi,avg}}$)"
        ),
    )

    # -------------------------------------------------------------------------
    # Weighted
    #
    # Every event contributes weight_lumi.
    # -------------------------------------------------------------------------

    make_mass_plot_weighted(
        signal_mass,
        background_mass,
        signal_weights,
        background_weights,
        OUT_DIR / "mllg_tthlep_weighted.png",
        title=(
            r"$m_{\ell\ell\gamma}$ shape — "
            r"ttHlep (weighted)"
        ),
    )
# =============================================================================
# 13. Main
# =============================================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load both scored directories
    # -------------------------------------------------------------------------

    print("\nLoading fold0 scored directory...")
    signal_fold0, background_fold0, background_parts_fold0 = load_all(
        DATA_DIR_FOLD0
    )

    print("\nLoading fold1 scored directory...")
    signal_fold1, background_fold1, background_parts_fold1 = load_all(
        DATA_DIR_FOLD1
    )

    # -------------------------------------------------------------------------
    # Build true OOF SR samples
    #
    # fold0 model -> parity 0
    # fold1 model -> parity 1
    # -------------------------------------------------------------------------

    signal_oof = build_oof_sample(
        signal_fold0,
        signal_fold1,
        oof_sr_mask,
    )

    background_oof = build_oof_sample(
        background_fold0,
        background_fold1,
        oof_sr_mask,
    )

    # Print fold-by-fold sanity check
    signal_fold0_test = apply_mask(
        signal_fold0,
        oof_sr_mask(
            signal_fold0,
            FOLD0_OOF_PARITY,
        ),
    )

    background_fold0_test = apply_mask(
        background_fold0,
        oof_sr_mask(
            background_fold0,
            FOLD0_OOF_PARITY,
        ),
    )

    signal_fold1_test = apply_mask(
        signal_fold1,
        oof_sr_mask(
            signal_fold1,
            FOLD1_OOF_PARITY,
        ),
    )

    background_fold1_test = apply_mask(
        background_fold1,
        oof_sr_mask(
            background_fold1,
            FOLD1_OOF_PARITY,
        ),
    )

    print("\n" + "=" * 70)
    print("OOF SANITY CHECK")
    print("=" * 70)

    print(
        f"fold0 model -> parity {FOLD0_OOF_PARITY}: "
        f"S={np.sum(signal_fold0_test[WEIGHT]):.6f}, "
        f"B={np.sum(background_fold0_test[WEIGHT]):.6f}"
    )

    print(
        f"fold1 model -> parity {FOLD1_OOF_PARITY}: "
        f"S={np.sum(signal_fold1_test[WEIGHT]):.6f}, "
        f"B={np.sum(background_fold1_test[WEIGHT]):.6f}"
    )

    print(
        f"OOF total: "
        f"S={np.sum(signal_oof[WEIGHT]):.6f}, "
        f"B={np.sum(background_oof[WEIGHT]):.6f}"
    )

    # -------------------------------------------------------------------------
    # Data sideband
    # Use fold0/Data directly; Data is not OOF-split.
    # -------------------------------------------------------------------------

    data_weights, data_scores, data_tags = load_data_sideband(
        DATA_DIR
    )

    # -------------------------------------------------------------------------
    # Cut-based
    # -------------------------------------------------------------------------

    cut_results = calculate_cutbased(
        signal_oof,
        background_oof,
        data_tags,
    )

    print_cutbased_table(
        cut_results
    )

    z_cut, sigma_cut = combined_z(
        cut_results
    )

    # -------------------------------------------------------------------------
    # GNN Bayesian optimization
    # -------------------------------------------------------------------------

    c1_bayes, c2_bayes, bayes_results = optimize_bayesian(
        signal_oof,
        background_oof,
        data_weights,
        data_scores,
    )

    print_gnn_results(
        "GNN BAYESIAN",
        bayes_results,
        c1_bayes,
        c2_bayes,
    )

    # -------------------------------------------------------------------------
    # Mass plots:
    #   weighted + unweighted
    # -------------------------------------------------------------------------

    plot_gnn_high_mass(
        signal_fold0,
        background_fold0,
        signal_fold1,
        background_fold1,
        c1_bayes,
    )

    plot_tthlep_mass(
        signal_fold0,
        background_fold0,
        signal_fold1,
        background_fold1,
    )

    # -------------------------------------------------------------------------
    # Background composition
    # -------------------------------------------------------------------------

    print_high_background_breakdown(
        background_parts_fold0,
        background_parts_fold1,
        c1_bayes,
    )

    print_high_background_avgweight_diagnostics(
        background_parts_fold0,
        background_parts_fold1,
        c1_bayes,
    )

    # -------------------------------------------------------------------------
    # Grid verification
    # -------------------------------------------------------------------------

    (
        grid_results,
        z_grid,
        c1_grid,
        c2_grid,
    ) = scan_gnn(
        signal_oof,
        background_oof,
        data_weights,
        data_scores,
    )

    if grid_results is not None:
        print_gnn_results(
            "GNN GRID",
            grid_results,
            c1_grid,
            c2_grid,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    z_bayes, sigma_bayes = combined_z(
        bayes_results
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Cut-based : "
        f"Z = {z_cut:.6f} ± {sigma_cut:.6f}"
    )

    print(
        f"GNN Bayes : "
        f"Z = {z_bayes:.6f} ± {sigma_bayes:.6f} "
        f"(c1={c1_bayes:.6f}, c2={c2_bayes:.6f})"
    )

    if z_cut > 0.0:
        ratio = z_bayes / z_cut

        print(
            f"Bayes/Cut : "
            f"{ratio:.4f} "
            f"({(ratio - 1.0) * 100:+.2f}%)"
        )

    if grid_results is not None:
        _, sigma_grid = combined_z(
            grid_results
        )

        print(
            f"GNN Grid  : "
            f"Z = {z_grid:.6f} ± {sigma_grid:.6f} "
            f"(c1={c1_grid:.6f}, c2={c2_grid:.6f})"
        )


if __name__ == "__main__":
    main()
