#!/usr/bin/env python3
"""
significance_4class_v3_cr_4cat.py

4-class, 4-fold OOF significance study for the v3 (Mll80to100) HZg analysis,
using the method-2 (CR_lowIDMVA data-driven) background estimate.

NEW scheme (per skills/.../Z_workflow.txt):
  - ALL Higgs processes are Signal: ttH + VH + ggF + VBF (merged yield).
  - Events are split into 4 MUTUALLY EXCLUSIVE categories by the GNN argmax +
    3 cut points (c_ttH, c_VH, c_ggFVBF), NOT by source_tree_tag and NOT by
    High/Medium/Low score binning.
  - Each category (incl. Other) gets its own S, alpha, B, Asimov Z.
  - Z_fold = sqrt(Z_ttH^2 + Z_VH^2 + Z_ggFVBF^2 + Z_Other^2).
  - Z_final = sqrt(sum over 4 folds of Z_fold_test^2).

Label encoding (verified from root_gnn_dgl/configs/hzg/ttH_4class_4fold_fold0.yaml
and confirmed by argmax on known-MC samples):
    0 = bkg   (-> p_Other, softmax[:,0])
    1 = ttH   (-> p_ttH,   softmax[:,1])
    2 = VH    (-> p_VH,    softmax[:,2])
    3 = ggF_VBF (ggF+VBF merged in training, -> p_ggFVBF, softmax[:,3])

Category assignment (identical rule for Signal MC / nominal Data / CR Data):
    kmax = argmax(p_bkg, p_ttH, p_VH, p_ggFVBF)
    ttH-enriched    if kmax==1 and p_ttH    > c_ttH
    VH-enriched     if kmax==2 and p_VH     > c_VH
    ggFVBF-enriched if kmax==3 and p_ggFVBF > c_ggFVBF
    Other           otherwise  (kmax==0, or failed the corresponding cut)

SR/SB are PURELY by mass (no is_SR, no source_tree_tag):
    SR = m_llg in (120,130)
    SB = m_llg in [110,120) U [130,180]

Method-2 background per category i:
    alpha_i = N_Data_SB,i / N_CR_SB,i
    B_i     = alpha_i * N_CR_SR,i
    (sigma_B/B)^2 = 1/N_Data_SB + 1/N_CR_SB + 1/N_CR_SR

Usage:
    python significance_4class_v3_cr_4cat.py
    python significance_4class_v3_cr_4cat.py --no-optuna --c-tth 0.3 --c-vh 0.3 --c-ggfvbf 0.3
    python significance_4class_v3_cr_4cat.py --trials 200
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_4class_v3_cr_4cat"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"
CR_SCORED_BASE = REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold"

# fold parity (same as significance_4fold.py / significance_4class_v3_cr.py)
TEST_PARITY  = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

TREE_NOM = "merged_SR_Sideband"
TREE_CR  = "merged_CR_lowIDMVA"
WEIGHT   = "weight_lumi"
MASS     = "m_llg"
SCORE    = "ttH_gnn_score"   # vector<float>(4): [p_bkg, p_ttH, p_VH, p_ggFVBF]
FOLDVAR  = "photon_eta_fold"

# mass windows
MASS_LO = 120.0
MASS_HI = 130.0
SB_LO1, SB_HI1 = 50.0, 120.0
SB_LO2, SB_HI2 = 130.0, 250.0

# label -> score column index (verified from training config)
COL_BKG, COL_TTH, COL_VH, COL_GGFVBF = 0, 1, 2, 3

# 4 categories: 0=Other, 1=ttH, 2=VH, 3=ggFVBF
CATS = ["Other", "ttH", "VH", "ggFVBF"]
CAT_OTHER, CAT_TTH, CAT_VH, CAT_GGFVBF = 0, 1, 2, 3

# all 4 signal processes (merged yield); ggF and VBF share label 3
SIGNAL_DIRS = ["Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF"]

ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

# Optuna: 3 cut points, jointly optimised
CUT_MIN, CUT_MAX = 0.10, 0.90
N_TRIALS         = 1000
N_STARTUP_TRIALS = 50
RANDOM_STATE     = 12345

# guards
MIN_N_CR_SB = 1   # require >=1 CR SB event or the cat's alpha is undefined


# =============================================================================
# 2. Statistics (same as significance_4class_v3_cr.py)
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
    return {"sum": sumw, "err": float(np.sqrt(sumw2)), "n": int(w.size),
            "nneg": int(np.sum(w < 0.0)), "neff": float(neff)}


def z_uncertainty(s, b, sigma_s, sigma_b):
    z = asimov_z(s, b)
    if z <= 0.0:
        return 0.0, 0.0, 0.0
    log_term = np.log1p(s / b)
    dz_ds = log_term / z
    dz_db = (log_term - s / b) / z
    return (float(np.hypot(dz_ds * sigma_s, dz_db * sigma_b)),
            float(abs(dz_ds * sigma_s)),
            float(abs(dz_db * sigma_b)))


def combined_z(results):
    """Quadrature combination over the 4 categories."""
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z = float(np.sqrt(z2))
    if z <= 0.0:
        return 0.0, 0.0
    sigma2 = sum(((r["Z"] / z) * r["sigma_Z"]) ** 2 for r in results.values())
    return z, float(np.sqrt(sigma2))


def sigma_b_transfer(n_data_sb, n_cr_sb, n_cr_sr):
    if n_cr_sb < MIN_N_CR_SB or n_data_sb <= 0:
        return 0.0, 0.0
    b = n_data_sb * n_cr_sr / n_cr_sb
    if b <= 0.0:
        return 0.0, 0.0
    var = (1.0 / n_data_sb) + (1.0 / n_cr_sb) + (1.0 / n_cr_sr if n_cr_sr > 0 else 0.0)
    return float(b), float(b * np.sqrt(var))


# =============================================================================
# 3. ROOT loading helpers
# =============================================================================

def _stack_scores_4d(sc):
    """vector<float>(4) branch -> (N,4) float32. Returns all 4 columns."""
    sc = np.asarray(sc)
    if sc.dtype == object or sc.ndim == 1:
        sc = np.stack([np.asarray(v, dtype=np.float32) for v in sc])
    return sc.astype(np.float32)   # (N,4): [p_bkg, p_ttH, p_VH, p_ggFVBF]


class EventSet:
    __slots__ = ("mass", "weight", "score", "foldvar")
    def __init__(self, mass, weight, score, foldvar):
        self.mass = mass
        self.weight = weight
        self.score = score    # (N,4)
        self.foldvar = foldvar
    @property
    def n(self):
        return self.mass.size


def empty_set():
    return EventSet(
        np.array([], np.float64), np.array([], np.float64),
        np.empty((0, 4), np.float32), np.array([], np.int32),
    )


def _concat_eventsets(parts):
    parts = [p for p in parts if p.n > 0]
    if not parts:
        return empty_set()
    return EventSet(
        np.concatenate([p.mass for p in parts]),
        np.concatenate([p.weight for p in parts]),
        np.concatenate([p.score for p in parts], axis=0),
        np.concatenate([p.foldvar for p in parts]),
    )


def _load_one_dir(base, subdir, tree_name, with_weight):
    """Load one subdir of a fold scored dir. Returns EventSet (all events, no mass cut)."""
    branches = [MASS, SCORE, FOLDVAR]
    if with_weight:
        branches.append(WEIGHT)
    pieces = []
    for era in ALL_ERAS:
        p = base / subdir / f"{era}.root"
        if not p.exists():
            continue
        try:
            with uproot.open(str(p)) as f:
                t = f[tree_name]
                arr = t.arrays(branches, library="np")
        except Exception as exc:
            print(f"WARNING: cannot read {p}: {exc}", file=sys.stderr)
            continue
        w = np.asarray(arr[WEIGHT], np.float64) if with_weight else np.ones(arr[MASS].shape, np.float64)
        pieces.append(EventSet(
            np.asarray(arr[MASS], np.float64),
            w,
            _stack_scores_4d(arr[SCORE]),
            np.asarray(arr[FOLDVAR], np.int32),
        ))
    return _concat_eventsets(pieces)


def load_signal_all(fold):
    """Load all 4 Higgs signal processes (ttH+VH+ggF+VBF) for one fold, merged."""
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    return _concat_eventsets([_load_one_dir(base, d, TREE_NOM, True) for d in SIGNAL_DIRS])


def load_data_sideband(fold):
    """nominal Data, SB mass window (pure mass). weight=1.0 for Data."""
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    es = _load_one_dir(base, "Data/Data", TREE_NOM, False)
    if es.n == 0:
        return es
    mask = np.isfinite(es.mass) & in_sb_mass(es.mass)
    return EventSet(es.mass[mask], es.weight[mask], es.score[mask], es.foldvar[mask])


def load_cr(fold):
    """CR Data, split into SR / SB by pure mass. weight=1.0 for Data."""
    base = CR_SCORED_BASE.parent / f"{CR_SCORED_BASE.name}{fold}"
    es = _load_one_dir(base, "Data/Data", TREE_CR, False)
    if es.n == 0:
        return empty_set(), empty_set()
    finite = np.isfinite(es.mass)
    sr = finite & in_sr_mass(es.mass)
    sb = finite & in_sb_mass(es.mass)
    return (EventSet(es.mass[sr], es.weight[sr], es.score[sr], es.foldvar[sr]),
            EventSet(es.mass[sb], es.weight[sb], es.score[sb], es.foldvar[sb]))


def in_sr_mass(m):
    return (m > MASS_LO) & (m < MASS_HI)

def in_sb_mass(m):
    return ((m >= SB_LO1) & (m < SB_HI1)) | ((m >= SB_LO2) & (m <= SB_HI2))


def select_parity(es, parity):
    if es.n == 0:
        return es
    m = (es.foldvar % 4) == parity
    return EventSet(es.mass[m], es.weight[m], es.score[m], es.foldvar[m])


# =============================================================================
# 4. Category assignment (argmax + 3 cuts), identical for all samples
# =============================================================================

def assign_cat(score4d, c_tth, c_vh, c_ggfvbf):
    """score4d: (N,4) = [p_bkg, p_ttH, p_VH, p_ggFVBF].
    Returns int32 array: 0=Other, 1=ttH, 2=VH, 3=ggFVBF."""
    if score4d.shape[0] == 0:
        return np.array([], np.int32)
    kmax = np.argmax(score4d, axis=1)
    cat = np.zeros(score4d.shape[0], dtype=np.int32)   # default Other
    cat[(kmax == COL_TTH)    & (score4d[:, COL_TTH]    > c_tth)]    = CAT_TTH
    cat[(kmax == COL_VH)     & (score4d[:, COL_VH]     > c_vh)]     = CAT_VH
    cat[(kmax == COL_GGFVBF) & (score4d[:, COL_GGFVBF] > c_ggfvbf)] = CAT_GGFVBF
    return cat


# =============================================================================
# 5. 4-category evaluation with method-2 background
# =============================================================================

def evaluate_bins(sig, data_sb, cr_sr, cr_sb, c_tth, c_vh, c_ggfvbf):
    """Evaluate 4 categories. Returns {cat: result}.

    S  = (ttH+VH+ggF+VBF) weighted yield in SR, per category.
    B  = N_Data_SB * N_CR_SR / N_CR_SB, per category (pure mass, unweighted counts).
    """
    sig_cat  = assign_cat(sig.score,     c_tth, c_vh, c_ggfvbf)
    data_cat = assign_cat(data_sb.score, c_tth, c_vh, c_ggfvbf) if data_sb.n else np.array([], np.int32)
    cs_cat   = assign_cat(cr_sr.score,   c_tth, c_vh, c_ggfvbf) if cr_sr.n else np.array([], np.int32)
    cb_cat   = assign_cat(cr_sb.score,   c_tth, c_vh, c_ggfvbf) if cr_sb.n else np.array([], np.int32)

    sig_mass = sig.mass
    sig_sr = (sig.n > 0) & np.isfinite(sig_mass) & in_sr_mass(sig_mass)

    results = {}
    for ci, cat in enumerate(CATS):
        sig_sel = sig_sr & (sig_cat == ci)
        S = yield_info(sig.weight[sig_sel])

        n_data_sb = int(np.sum((data_cat == ci) & in_sb_mass(data_sb.mass))) if data_sb.n else 0
        n_cr_sr = int(np.sum((cs_cat == ci) & in_sr_mass(cr_sr.mass))) if cr_sr.n else 0
        n_cr_sb = int(np.sum((cb_cat == ci) & in_sb_mass(cr_sb.mass))) if cr_sb.n else 0

        B_val, sigma_b = sigma_b_transfer(n_data_sb, n_cr_sb, n_cr_sr)
        sigma_s = S["err"]

        valid = (B_val > 0.0) and (n_cr_sb >= MIN_N_CR_SB) and (S["sum"] >= 0.0)
        z = asimov_z(S["sum"], B_val) if valid else 0.0
        sigma_z, sigma_z_s, sigma_z_b = (
            z_uncertainty(S["sum"], B_val, sigma_s, sigma_b) if valid else (0.0, 0.0, 0.0)
        )

        results[cat] = {
            "S": S["sum"], "sigma_S": sigma_s,
            "B": B_val, "sigma_B": sigma_b,
            "alpha": (n_data_sb / n_cr_sb) if n_cr_sb >= MIN_N_CR_SB else float("nan"),
            "N_data_sb": n_data_sb, "N_cr_sr": n_cr_sr, "N_cr_sb": n_cr_sb,
            "Z": z, "sigma_Z": sigma_z,
            "sigma_Z_from_S": sigma_z_s, "sigma_Z_from_B": sigma_z_b,
            "valid": valid,
        }
    return results


def flatten(results):
    return results   # already {cat: result}


# =============================================================================
# 6. Printing
# =============================================================================

def print_bins(title, results, c_tth, c_vh, c_ggfvbf):
    flat = flatten(results)
    z, sigma_z = combined_z(flat)
    w = 120
    print("\n" + "=" * w)
    print(f"{title}: c_ttH={c_tth:.4f}, c_VH={c_vh:.4f}, c_ggFVBF={c_ggfvbf:.4f}")
    print("=" * w)
    hdr = (f"{'cat':9s} "
           f"{'S':>10}{'sigS':>9}"
           f"{'NdatSB':>8}{'NcrSB':>7}{'NcrSR':>7}"
           f"{'alpha':>8}{'B':>10}{'sigB':>9}"
           f"{'Z':>9}{'sigZ':>9}")
    print(hdr); print("-" * len(hdr))
    for cat in CATS:
        r = results[cat]
        print(f"{cat:9s} "
              f"{r['S']:>10.4f}{r['sigma_S']:>9.4f}"
              f"{r['N_data_sb']:>8d}{r['N_cr_sb']:>7d}{r['N_cr_sr']:>7d}"
              f"{r['alpha']:>8.3f}{r['B']:>10.3f}{r['sigma_B']:>9.3f}"
              f"{r['Z']:>9.4f}{r['sigma_Z']:>9.4f}")
    print("-" * len(hdr))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}  (sqrt of sum of {sum(1 for r in flat.values() if r['valid'])} valid cats)")


# =============================================================================
# 7. Bayesian optimisation (val -> 3 cuts)
# =============================================================================

def optimize_bayesian(sig_val, data_sb_val, cr_sr_val, cr_sb_val, label=""):
    if optuna is None:
        raise RuntimeError("optuna not available; pass --no-optuna with --c-tth/--c-vh/--c-ggfvbf")

    def objective(trial):
        c_tth = trial.suggest_float("c_tth", CUT_MIN, CUT_MAX)
        c_vh  = trial.suggest_float("c_vh",  CUT_MIN, CUT_MAX)
        c_ggfvbf = trial.suggest_float("c_ggfvbf", CUT_MIN, CUT_MAX)
        results = evaluate_bins(sig_val, data_sb_val, cr_sr_val, cr_sb_val, c_tth, c_vh, c_ggfvbf)
        z, _ = combined_z(results)
        return z if z > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nBayesian optimisation {label}: {N_TRIALS} trials (3 cuts) ...")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    best = max(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value and t.value > 0),
        key=lambda t: t.value, default=None,
    )
    if best is None:
        raise RuntimeError(f"No valid trial for {label}; try --no-optuna --c-tth/--c-vh/--c-ggfvbf")
    return (float(best.params["c_tth"]), float(best.params["c_vh"]), float(best.params["c_ggfvbf"]))


# =============================================================================
# 8. Main
# =============================================================================

def main():
    global N_TRIALS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-optuna", action="store_true",
                    help="skip optimisation; use --c-tth/--c-vh/--c-ggfvbf for all folds")
    ap.add_argument("--c-tth", type=float, default=0.3)
    ap.add_argument("--c-vh", type=float, default=0.3)
    ap.add_argument("--c-ggfvbf", type=float, default=0.3)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    N_TRIALS = args.trials

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("4-CATEGORY SCHEME: all-Higgs signal, GNN argmax + 3 cuts, CR method-2 background")
    print("=" * 90)
    print(f"Signal: {'+'.join(SIGNAL_DIRS)}  (merged yield, weight_lumi)")
    print(f"Categories: {CATS}  (Other = bkg-argmax or failed cut)")
    print(f"SR=(120,130)  SB=[110,120)U[130,180]  (pure mass, no is_SR/tag)")
    if args.no_optuna:
        print(f"[no-optuna] c_ttH={args.c_tth} c_VH={args.c_vh} c_ggFVBF={args.c_ggfvbf}")

    fold_results = {}
    for fold_idx in range(4):
        val_p  = VALID_PARITY[fold_idx]
        test_p = TEST_PARITY[fold_idx]
        print(f"\n{'='*70}\nFOLD {fold_idx}  val parity={val_p}  test parity={test_p}\n{'='*70}")

        print("  Loading signal (4 processes)...")
        sig_all = load_signal_all(fold_idx)
        print(f"    N_sig={sig_all.n}")
        print("  Loading nominal Data SB...")
        data_sb_all = load_data_sideband(fold_idx)
        print(f"    N_data_sb={data_sb_all.n}")
        print("  Loading CR Data (SR + SB)...")
        cr_sr_all, cr_sb_all = load_cr(fold_idx)
        print(f"    N_cr_sr={cr_sr_all.n}  N_cr_sb={cr_sb_all.n}")

        sig_val, sig_test = select_parity(sig_all, val_p), select_parity(sig_all, test_p)
        data_sb_val, data_sb_test = select_parity(data_sb_all, val_p), select_parity(data_sb_all, test_p)
        cr_sr_val, cr_sr_test = select_parity(cr_sr_all, val_p), select_parity(cr_sr_all, test_p)
        cr_sb_val, cr_sb_test = select_parity(cr_sb_all, val_p), select_parity(cr_sb_all, test_p)

        print(f"  val : sig={sig_val.n} dataSB={data_sb_val.n} crSR={cr_sr_val.n} crSB={cr_sb_val.n}")
        print(f"  test: sig={sig_test.n} dataSB={data_sb_test.n} crSR={cr_sr_test.n} crSB={cr_sb_test.n}")

        if args.no_optuna:
            c_tth, c_vh, c_ggfvbf = args.c_tth, args.c_vh, args.c_ggfvbf
            print(f"  [skip optuna] c_ttH={c_tth} c_VH={c_vh} c_ggFVBF={c_ggfvbf}")
        else:
            c_tth, c_vh, c_ggfvbf = optimize_bayesian(
                sig_val, data_sb_val, cr_sr_val, cr_sb_val, label=f"fold{fold_idx} val")

        val_results = evaluate_bins(sig_val, data_sb_val, cr_sr_val, cr_sb_val, c_tth, c_vh, c_ggfvbf)
        print_bins(f"FOLD {fold_idx} VALIDATION", val_results, c_tth, c_vh, c_ggfvbf)
        z_val, _ = combined_z(val_results)

        test_results = evaluate_bins(sig_test, data_sb_test, cr_sr_test, cr_sb_test, c_tth, c_vh, c_ggfvbf)
        print_bins(f"FOLD {fold_idx} TEST", test_results, c_tth, c_vh, c_ggfvbf)
        z_test, sigma_z_test = combined_z(test_results)

        fold_results[fold_idx] = {
            "c_tth": c_tth, "c_vh": c_vh, "c_ggfvbf": c_ggfvbf,
            "val_results": val_results, "test_results": test_results,
            "Z_val": z_val, "Z_test": z_test, "sigma_Z_test": sigma_z_test,
        }

    print("\n" + "=" * 80)
    print("PER-FOLD SUMMARY")
    print("=" * 80)
    print(f"{'Fold':<6}{'c_ttH':>9}{'c_VH':>9}{'c_ggFVBF':>10}{'Z_val':>12}{'Z_test':>14}{'sigZ_test':>12}")
    print("-" * 70)
    z_sq = 0.0
    for k, r in fold_results.items():
        print(f"{k:<6}{r['c_tth']:>9.4f}{r['c_vh']:>9.4f}{r['c_ggfvbf']:>10.4f}"
              f"{r['Z_val']:>12.6f}{r['Z_test']:>14.6f}{r['sigma_Z_test']:>12.6f}")
        z_sq += r["Z_test"] ** 2
    z_final = float(np.sqrt(z_sq))
    sigma_sq = 0.0
    for r in fold_results.values():
        if z_final > 0.0 and r["Z_test"] > 0.0:
            sigma_sq += ((r["Z_test"] / z_final) * r["sigma_Z_test"]) ** 2
    sigma_final = float(np.sqrt(sigma_sq))

    print("\n" + "=" * 70)
    print("FINAL RESULT (4-category, method-2 CR transfer-factor background)")
    print("=" * 70)
    print(f"4-fold combined Z = {z_final:.6f} ± {sigma_final:.6f}")
    print(f"  = sqrt(" + " + ".join(f"{r['Z_test']:.4f}^2" for r in fold_results.values()) + ")")

    rep = OUT_DIR / "final_z.txt"
    with open(rep, "w") as fh:
        fh.write("4-category method-2 CR transfer-factor significance\n")
        fh.write(f"combined Z = {z_final:.6f} ± {sigma_final:.6f}\n\n")
        fh.write(f"{'fold':<6}{'c_ttH':>9}{'c_VH':>9}{'c_ggFVBF':>10}{'Z_val':>12}{'Z_test':>14}{'sigZ':>12}\n")
        for k, r in fold_results.items():
            fh.write(f"{k:<6}{r['c_tth']:>9.4f}{r['c_vh']:>9.4f}{r['c_ggfvbf']:>10.4f}"
                     f"{r['Z_val']:>12.6f}{r['Z_test']:>14.6f}{r['sigma_Z_test']:>12.6f}\n")
    print(f"\nReport: {rep}")


if __name__ == "__main__":
    main()
