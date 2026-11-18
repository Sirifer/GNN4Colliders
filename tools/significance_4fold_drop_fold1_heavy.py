#!/usr/bin/env python3
"""
4-fold OOF significance study for ttH pt_only.

For each fold k:
  - validation set  = events with (photon_eta_fold % 4) == VALID_PARITY[k]
                      → used to optimise GNN category boundaries
  - test set        = events with (photon_eta_fold % 4) == TEST_PARITY[k]
                      → boundaries applied here to produce the final Z_k

Fold assignments (from config.yaml):
  fold0: test=[0], val=[1], train=[2,3]
  fold1: test=[1], val=[2], train=[3,0]
  fold2: test=[2], val=[3], train=[0,1]
  fold3: test=[3], val=[0], train=[1,2]

Data sideband: scored by the 2-fold fold0 model (only available source),
used as a stability constraint during fold0 boundary optimisation only.

Final combined Z = sqrt( Z0^2 + Z1^2 + Z2^2 + Z3^2 )
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
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results_4fold_pt_only"
)

FOLD_DIRS = [
    Path(
        f"/scratchfs2/cms/huangsuyun/GNN4Colliders"
        f"/root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{i}"
    )
    for i in range(4)
]

# Data sideband: 2-fold fold0 scored directory (the only place with Data).
# These scores come from the 2-fold model, so they are used only as a loose
# stability constraint for fold0's boundary optimisation.
DATA_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders"
    "/root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
) / "Data"

# Fold parity assignments
TEST_PARITY  = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

TREE     = "merged_SR_Sideband"
WEIGHT   = "weight_lumi"
WEIGHT_AVG = "weight_lumi_avg"
MASS     = "m_llg"
SCORE    = "ttH_gnn_score"
IS_SR    = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR  = "photon_eta_fold"

BRANCHES = [WEIGHT, WEIGHT_AVG, MASS, SCORE, IS_SR, TREE_TAG, FOLDVAR]
DATA_BRANCHES = [WEIGHT, SCORE, IS_SR, TREE_TAG]
# Optional event-id branches, loaded when present, used only to tighten the
# heavy-event drop match. Their absence is harmless — the drop falls back to a
# (weight, mass, tag) physics fingerprint, which is unique for this event.
ID_BRANCHES = ["event", "run"]

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

C1_VALUES = np.arange(0.10, 0.951, 0.01)
C2_VALUES = np.append(np.arange(C2_MIN, C2_MAX, 0.005), C2_MAX)

MIN_B_NEFF             = 0.0
MIN_HIGH_SR_TOTAL_YIELD = 0.0
MIN_HIGH_DATA_SB_YIELD  = 0   # require > 0 data sideband events in high cat


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
    """Quadrature combination of per-category Z values."""
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z  = float(np.sqrt(z2))
    if z <= 0.0:
        return 0.0, 0.0
    sigma2 = sum(((r["Z"] / z) * r["sigma_Z"]) ** 2 for r in results.values())
    return z, float(np.sqrt(sigma2))


# =============================================================================
# 3. ROOT helpers
# =============================================================================

def empty_arrays():
    base = {b: np.array([]) for b in BRANCHES}
    base.update({b: np.array([]) for b in ID_BRANCHES})
    return base


def concatenate_arrs(arrays):
    out = {}
    for b in BRANCHES:
        pieces = [a[b] for a in arrays if a[b].size > 0]
        out[b] = np.concatenate(pieces) if pieces else np.array([])
    # ID branches may be present in only a subset of files; union what exists.
    for b in ID_BRANCHES:
        pieces = []
        for a in arrays:
            v = a.get(b)
            if v is not None and getattr(v, "size", 0) > 0:
                pieces.append(v)
        out[b] = np.concatenate(pieces) if pieces else np.array([])
    return out


def apply_mask(arr, mask):
    out = {b: arr[b][mask] for b in BRANCHES}
    # Carry id branches through the mask if they are aligned to the event count.
    n = arr[WEIGHT].size
    for b in ID_BRANCHES:
        v = arr.get(b)
        if v is not None and v.size == n:
            out[b] = v[mask]
        else:
            out[b] = np.array([])
    return out


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
            # Load core branches plus any id branches that exist in this file.
            load = list(BRANCHES)
            for b in ID_BRANCHES:
                if b in tree.keys() and b not in load:
                    load.append(b)
            arr = tree.arrays(load, library="np")
            out = empty_arrays()
            for b in load:
                out[b] = arr[b]
            return out
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
    n_used = 0
    print(f"\nLoading Data sideband from {DATA_DIR} ...")
    for path in files:
        try:
            with uproot.open(str(path)) as f:
                if TREE not in f:
                    continue
                tree = f[TREE]
                missing = [b for b in DATA_BRANCHES if b not in tree.keys()]
                if missing:
                    print(f"WARNING: {path} missing branches: {missing}", file=sys.stderr)
                    continue
                arr = tree.arrays(DATA_BRANCHES, library="np")
        except Exception as exc:
            print(f"WARNING: {path}: {exc}", file=sys.stderr)
            continue
        mask = (arr[IS_SR] == 0) & np.isfinite(arr[WEIGHT]) & np.isfinite(arr[SCORE])
        if np.any(mask):
            weights.append(np.asarray(arr[WEIGHT][mask], dtype=float))
            scores.append(np.asarray(arr[SCORE][mask],  dtype=float))
            tags.append(  np.asarray(arr[TREE_TAG][mask]))
            n_used += 1
    if not weights:
        raise RuntimeError("No usable Data sideband events (is_SR == 0).")
    weights = np.concatenate(weights)
    scores  = np.concatenate(scores)
    tags    = np.concatenate(tags)
    print(
        f"Data sideband: files={n_used}, "
        f"N={weights.size}, yield={np.sum(weights):.4f}"
    )
    return weights, scores, tags


# =============================================================================
# 4. Event selection
# =============================================================================

# --- Diagnostic drop for the TT/2024 heavy event (weight ~ 0.7448) ---
# Event fingerprint (from the 2026-10-27 log + find_fold1_heavy_eventid.py):
#   weight_lumi = 0.744848, m_llg = 125.9755, source_tree_tag = 0 (ttHhad),
#   run = 1, event = 82508110, photon_eta_fold = -6783  (=> parity = -6783 % 4 = 1)
#
# This single event dominates fold1 test high B (B = 0.7456, σB = 0.7448).
# Because parity == 1, it appears in:
#   - fold0 as the VALIDATION set (val parity = 1)  -> it steers fold0's boundary
#   - fold1 as the TEST set      (test parity = 1)  -> it drives fold1's final Z
# It is NOT in fold2/fold3 val/test (those use parities 3/0 and 2/3); it sits in
# their training set, which is frozen (the GNN is already trained).
#
# Match is by the physics-invariant quantities (weight, mass, tag) ONLY — NOT
# score. Score is model-dependent and differs across fold directories, so a
# score window would miss the event in fold0 even though it is the same event.
# weight ~ 0.7448 is extreme and unique across all background samples, so the
# (weight, mass, tag) fingerprint is unambiguous.
#
# This variant is a DIAGNOSTIC / robustness probe only. If the weight is
# correct (verified via find_fold1_heavy_eventid.py: unique, run=1, event=82508110),
# the event is real background and must NOT be removed from a final result —
# removing it would bias the result. Here we drop it only to measure how much
# fold1 / the combined Z depends on it.
DROP_HEAVY = True
DROP_TAG     = 0          # ttHhad
DROP_W_LO, DROP_W_HI = 0.74, 0.75
DROP_M_LO, DROP_M_HI = 125.0, 126.5
# Match the documented event id when present, as an extra guard. We also keep a
# fallback to the (weight, mass, tag) fingerprint because some fold directories
# may not carry an 'event' branch.
DROP_RUN   = 1
DROP_EVENT = 82508110


def _heavy_mask(arr):
    """Boolean mask True for the single heavy TT/2024 event in `arr`."""
    if arr[WEIGHT].size == 0:
        return np.zeros(0, dtype=bool)
    base = (
        (arr[WEIGHT]    > DROP_W_LO) & (arr[WEIGHT]    < DROP_W_HI)
        & (arr[MASS]     > DROP_M_LO) & (arr[MASS]     < DROP_M_HI)
        & (arr[TREE_TAG] == DROP_TAG)
    )
    # Tighten with the exact event id if that branch is present and aligned.
    if "event" in arr and arr["event"].size == arr[WEIGHT].size:
        ev = np.asarray(arr["event"])
        id_match = ev == DROP_EVENT
        if "run" in arr and arr["run"].size == arr[WEIGHT].size:
            id_match &= np.asarray(arr["run"]) == DROP_RUN
        # Only tighten — never loosen. If the id branch is junk, keep the
        # physics fingerprint match.
        if np.any(id_match):
            return base & id_match
    return base


def apply_drop(arr, parity):
    """Drop the heavy event wherever it appears (any fold, any parity).

    The event has parity == 1, so it only actually lands in fold0 val and
    fold1 test. This is applied to every fold's selected arrays so that, if the
    same physics event somehow entered a different parity bucket (it cannot,
    but defensively), it would still be caught.
    """
    if not DROP_HEAVY:
        return arr
    heavy = _heavy_mask(arr)
    n_drop = int(np.sum(heavy))
    if n_drop == 0:
        return arr
    keep = ~heavy
    print(f"  [drop] parity={parity}: dropping {n_drop} heavy event(s) "
          f"(weight={arr[WEIGHT][heavy][0]:.4f}, mass={arr[MASS][heavy][0]:.2f}, "
          f"tag={int(arr[TREE_TAG][heavy][0])})"
          + (f", run={int(arr['run'][heavy][0])}, event={int(arr['event'][heavy][0])}"
             if ("run" in arr and arr["run"].size) and ("event" in arr and arr["event"].size) else ""))
    return apply_mask(arr, keep)


def select_sr_parity(arr, parity):
    """SR events in the mass window belonging to the given parity bucket."""
    mask = (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
        & ((arr[FOLDVAR] % 4) == parity)
    )
    out = apply_mask(arr, mask)
    out = apply_drop(out, parity)
    return out


# =============================================================================
# 5. 3-category evaluation
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

        sigma_z, sigma_z_s, sigma_z_b = z_uncertainty(
            S["sum"], B["sum"], S["err"], B["err"]
        )
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
# 6. Cut-based significance
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
# 7. Bayesian optimisation
# =============================================================================

def optimize_bayesian(
    signal, background,
    data_weights, data_scores,
    check_data_sb=False,
    label="",
):
    def objective(trial):
        c1     = trial.suggest_float("c1", C1_MIN, C1_MAX)
        c2_low = max(C2_MIN, c1 + MIN_GAP)
        if c2_low >= C2_MAX:
            return -1.0
        c2 = trial.suggest_float("c2", c2_low, C2_MAX)
        results, z, valid = evaluate_3cat(
            signal, background,
            data_weights, data_scores,
            c1, c2,
            check_data_sb=check_data_sb,
        )
        trial.set_user_attr("valid", valid)
        return z if valid else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, n_startup_trials=N_STARTUP_TRIALS
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nBayesian optimisation {label}: {N_TRIALS} trials ...")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    valid_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("valid", False)
    ]
    if not valid_trials:
        raise RuntimeError(f"No valid Bayesian trial for {label}.")
    best = max(valid_trials, key=lambda t: t.value)
    c1   = float(best.params["c1"])
    c2   = float(best.params["c2"])
    results, _, _ = evaluate_3cat(
        signal, background,
        data_weights, data_scores,
        c1, c2,
        check_data_sb=check_data_sb,
    )
    return c1, c2, results


# =============================================================================
# 8. Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data sideband (2-fold fold0 scored; used only for fold0 constraint)
    data_weights, data_scores, data_tags = load_data_sideband()

    # Empty arrays for folds that don't use data constraint
    empty_w = np.array([])
    empty_s = np.array([])
    empty_t = np.array([], dtype=int)

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
        # Data tags available only from the 2-fold fold0 model; pass them for
        # all folds as a reference count (they are not split by parity here).
        cut_test = calculate_cutbased(sig_test, bkg_test, data_tags)
        print_cutbased_table(f"FOLD {fold_idx} TEST — CUT-BASED", cut_test)
        z_cut_test, _ = combined_z(cut_test)

        # --- Bayesian optimisation on validation set ---
        use_data = (fold_idx == 0)
        dw = data_weights if use_data else empty_w
        ds = data_scores  if use_data else empty_s

        c1_best, c2_best, val_res_bayes = optimize_bayesian(
            sig_val, bkg_val, dw, ds,
            check_data_sb=use_data,
            label=f"fold{fold_idx} val",
        )
        z_val_bayes, _ = combined_z(val_res_bayes)

        print_3cat(
            f"FOLD {fold_idx} VALIDATION (Bayesian)",
            val_res_bayes, c1_best, c2_best,
        )
        print(f"Fold {fold_idx} chosen boundary: c1={c1_best:.6f}, c2={c2_best:.6f}")

        # --- Evaluate GNN on test set with chosen boundary ---
        test_results, z_test, valid = evaluate_3cat(
            sig_test, bkg_test,
            empty_w, empty_s,
            c1_best, c2_best,
            check_data_sb=False,
        )

        if not valid:
            print(
                f"WARNING: fold {fold_idx} GNN test evaluation invalid "
                f"(B<=0 or other). Z_test=0.",
                file=sys.stderr,
            )
            z_test       = 0.0
            test_results = {}

        if test_results:
            print_3cat(f"FOLD {fold_idx} TEST — GNN", test_results, c1_best, c2_best)

        fold_results[fold_idx] = {
            "c1": c1_best, "c2": c2_best,
            "val_results":      val_res_bayes,
            "test_results":     test_results,
            "cut_test_results": cut_test,
            "Z_val":      z_val_bayes,
            "Z_test":     z_test,
            "Z_cut_test": z_cut_test,
        }

    # -------------------------------------------------------------------------
    # Combined Z from all 4 test sets
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

    # Error propagation for GNN combined Z
    sigma_sq = 0.0
    for r in fold_results.values():
        if r["test_results"] and z_gnn_final > 0.0:
            _, sigma_k = combined_z(r["test_results"])
            sigma_sq += ((r["Z_test"] / z_gnn_final) * sigma_k) ** 2
    sigma_gnn_final = float(np.sqrt(sigma_sq))

    # Error propagation for cut-based combined Z
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
        f"4-fold GNN  combined Z = {z_gnn_final:.6f} ± {sigma_gnn_final:.6f}"
        f"  = sqrt("
        + " + ".join(f"{r['Z_test']:.4f}^2" for r in fold_results.values())
        + ")"
    )
    print(
        f"4-fold cut-based     Z = {z_cut_final:.6f} ± {sigma_cut_final:.6f}"
        f"  = sqrt("
        + " + ".join(f"{r['Z_cut_test']:.4f}^2" for r in fold_results.values())
        + ")"
    )
    if z_cut_final > 0.0:
        ratio = z_gnn_final / z_cut_final
        print(f"GNN / cut-based = {ratio:.4f} ({(ratio - 1.0) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
