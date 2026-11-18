#!/usr/bin/env python3
"""
significance_4class_v3_cr.py

4-class, 4-fold OOF significance study for the v3 (Mll80to100) ttH analysis,
using the method-2 (CR_lowIDMVA data-driven) background estimate.

Based on tools/significance_4fold.py (the 2-class MC-counting-background version).
The statistical scaffolding -- OOF fold split, val/test boundary optimisation,
3-category score binning, Asimov Z, quadrature combination, uncertainty
propagation -- is preserved unchanged. Only the background estimate changes:
instead of B = weighted MC background yield, B is the CR transfer factor

    alpha(i,j) = N_Data_SB(i,j) / N_CR_SB(i,j)
    B(i,j)     = alpha(i,j) * N_CR_SR(i,j)

where i is one of the 5 analysis categories and j is one of the 3 score
regions (low/medium/high). Signal S still comes from ttH MC (OOF, weighted).

See tools/summary/hzg_4class_4fold_mll80to100/method2_z_calculation_principle.txt
for the full derivation and a per-item diff against significance_4fold.py.

Key multiclass note: the 4-class model emits a vector<float> score
[p_bkg, p_ttH, p_VH, p_ggFVBF]. The discriminant is p_ttH = softmax[:, 1].
uproot reads std::vector<float> as an object-dtype 1D array, so each score
branch is stacked into (N, 4) before taking column 1.

Key layout note: nominal and CR trees use different source_tree_tag encodings.
  nominal (merged_SR_Sideband): SR/SB alternate -- even=SR, odd=SB,
                                 tag//2 = category index. is_SR branch: 1=SR, 0=SB.
  CR      (merged_CR_lowIDMVA): region-grouped -- 0-4=CR SR, 5-9=CR SB,
                                 tag%5 = category index. No is_SR reliance;
                                 region decided by tag range.

Fold parity (same as significance_4fold.py):
  fold0: test=[0] val=[1] train=[2,3]
  fold1: test=[1] val=[2] train=[3,0]
  fold2: test=[2] val=[3] train=[0,1]
  fold3: test=[3] val=[0] train=[1,2]
Each event uses the score from fold directory == (photon_eta_fold % 4), i.e.
the model that did NOT train on that parity (OOF). nominal and CR both do this.

Combined Z = sqrt( Z0^2 + Z1^2 + Z2^2 + Z3^2 ), with each Z_k itself a
quadrature sum over 5 categories x 3 score regions = 15 bins.

Usage:
    python significance_4class_v3_cr.py
    python significance_4class_v3_cr.py --no-optuna --c1 0.3 --c2 0.8
    python significance_4class_v3_cr.py --trials 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

# optuna is optional at import time; only needed for boundary optimisation.
try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_4class_v3_cr"

# nominal MC scored (signal ttH; bkg MC kept only for a cut-based reference column)
MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
# nominal Data scored (method-2 denominator N_Data_SB)
DATA_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"
# CR Data scored (method-2 N_CR_SR and N_CR_SB)
CR_SCORED_BASE = REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold"

# --- fold parity (identical to significance_4fold.py) ---
TEST_PARITY  = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

# --- branches ---
TREE_NOM   = "merged_SR_Sideband"
TREE_CR    = "merged_CR_lowIDMVA"
WEIGHT     = "weight_lumi"        # MC per-event weight (Data=1.0)
MASS       = "m_llg"
SCORE      = "ttH_gnn_score"      # vector<float>(4); take [:,1]
IS_SR      = "is_SR"              # nominal only: 1=SR, 0=SB
TREE_TAG   = "source_tree_tag"
FOLDVAR    = "photon_eta_fold"

# --- mass windows (same as significance_4fold.py / diag_cr_counts_v2.py) ---
MASS_LO = 120.0
MASS_HI = 130.0
SB_LO1, SB_HI1 = 110.0, 120.0
SB_LO2, SB_HI2 = 130.0, 180.0

# --- 5 analysis categories ---
# nominal source_tree_tag: SR even {0,2,4,6,8}, SB = +1; tag//2 = cat index
NOM_SR_TAGS = np.array([0, 2, 4, 6, 8])      # ttHhad, ttHlep, untag, VH3l, VHpTmiss
NOM_SB_TAGS = NOM_SR_TAGS + 1
# CR source_tree_tag: SR {0,1,2,3,4}, SB {5,6,7,8,9}; tag%5 = cat index
CR_SR_TAGS  = np.array([0, 1, 2, 3, 4])
CR_SB_TAGS  = np.array([5, 6, 7, 8, 9])
CATS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]

# Run2+Run3, all 11 eras (v3; old script used 6 Run3 eras only)
ALL_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

SIGNAL_GROUP = ("Signal/ttH", ALL_ERAS)
# bkg MC groups kept only for the cut-based reference column (NOT used for B)
BKG_GROUPS = [
    ("Signal/VH",               ALL_ERAS),
    ("Signal/ggF",              ALL_ERAS),
    ("Signal/VBF",              ALL_ERAS),
    ("Bkg_MC/TT",               ALL_ERAS),
    ("Bkg_MC/DYGto2LG_10to100", ALL_ERAS),
    ("Bkg_MC/DYGto2LG_100to200",ALL_ERAS),
    ("Bkg_MC/DYGto2LG_200to400",ALL_ERAS),
    ("Bkg_MC/DYGto2LG_400to600",ALL_ERAS),
    ("Bkg_MC/DYGto2LG_600",     ALL_ERAS),
    ("Bkg_MC/DYJetsToLL_Stitch",ALL_ERAS),
]

# --- GNN boundary optimisation (same ranges as significance_4fold.py) ---
C1_MIN, C1_MAX = 0.10, 0.95
C2_MIN, C2_MAX = 0.20, 0.999
MIN_GAP        = 0.001
N_TRIALS         = 1000
N_STARTUP_TRIALS = 50
RANDOM_STATE     = 12345

# guards (method-2 specific: empty CR SB makes alpha undefined)
MIN_N_CR_SB = 1          # require >=1 CR SB event or the bin is invalid
MIN_N_CR_SR = 0          # CR SR may be 0 -> B=0 -> Z=0, fine
MIN_S_TOTAL = 0.5        # require some signal in the bin to be counted


# =============================================================================
# 2. Statistics (unchanged from significance_4fold.py)
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
    """Same form as significance_4fold.py; sigma_b source differs (see module doc)."""
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
    """Quadrature combination of per-bin Z values (unchanged)."""
    z2 = sum(r["Z"] ** 2 for r in results.values())
    z  = float(np.sqrt(z2))
    if z <= 0.0:
        return 0.0, 0.0
    sigma2 = sum(((r["Z"] / z) * r["sigma_Z"]) ** 2 for r in results.values())
    return z, float(np.sqrt(sigma2))


def sigma_b_transfer(n_data_sb, n_cr_sb, n_cr_sr):
    """Method-2 background uncertainty from CR transfer factor propagation.

    B = N_Data_SB * N_CR_SR / N_CR_SB  (Data weights = 1, so these are counts).
    (sigma_B / B)^2 = 1/N_Data_SB + 1/N_CR_SB + 1/N_CR_SR   (independent Poisson).
    Returns (B, sigma_B); B=0 if any denominator/numerator is degenerate.
    """
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

def _stack_score(sc):
    """vector<float> branch -> (N,4) float32; take [:,1] upstream.

    uproot reads std::vector<float> as an object-dtype 1D array (each entry a
    small ndarray). Stack into 2D. If already numeric 2D, pass through.
    """
    sc = np.asarray(sc)
    if sc.dtype == object or sc.ndim == 1:
        sc = np.stack([np.asarray(v, dtype=np.float32) for v in sc])
    return sc[:, 1].astype(np.float32)   # p_ttH


class EventSet:
    """Holds parallel arrays for one population (signal / data-sb / cr-sr / cr-sb)."""
    __slots__ = ("mass", "weight", "score", "tag", "foldvar")

    def __init__(self, mass, weight, score, tag, foldvar):
        self.mass = mass
        self.weight = weight
        self.score = score
        self.tag = tag
        self.foldvar = foldvar

    @property
    def n(self):
        return self.mass.size


def empty_set():
    return EventSet(
        np.array([], np.float64), np.array([], np.float64),
        np.array([], np.float32), np.array([], np.int32), np.array([], np.int32),
    )


def load_nominal_mc(fold, subdir, eras):
    """Load a nominal MC group (Signal/ttH etc.) from one fold scored dir.

    Returns EventSet (mass, weight, score, source_tree_tag, photon_eta_fold).
    Score is the OOF ttH score for THIS fold (events here all use fold `fold`'s
    model -- caller is responsible for parity selection downstream).
    """
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    pieces = []
    for era in eras:
        p = base / subdir / f"{era}.root"
        if not p.exists():
            continue
        try:
            with uproot.open(str(p)) as f:
                t = f[TREE_NOM]
                arr = t.arrays([MASS, WEIGHT, SCORE, TREE_TAG, FOLDVAR], library="np")
        except Exception as exc:
            print(f"WARNING: cannot read {p}: {exc}", file=sys.stderr)
            continue
        pieces.append((
            np.asarray(arr[MASS], np.float64),
            np.asarray(arr[WEIGHT], np.float64),
            _stack_score(arr[SCORE]),
            np.asarray(arr[TREE_TAG], np.int32),
            np.asarray(arr[FOLDVAR], np.int32),
        ))
    if not pieces:
        return empty_set()
    return EventSet(
        np.concatenate([x[0] for x in pieces]),
        np.concatenate([x[1] for x in pieces]),
        np.concatenate([x[2] for x in pieces]),
        np.concatenate([x[3] for x in pieces]),
        np.concatenate([x[4] for x in pieces]),
    )


def load_data_sideband(fold):
    """nominal Data sideband from fold `fold` scored dir.

    Method-2 denominator N_Data_SB. Selection is PURELY by mass:
    m_llg in the SB window [110,120) U [130,180].
    is_SR is NOT used -- it marks which analysis tree (SR/Sideband) an event
    came from, not whether m_llg is in the mass sideband (the original v3
    trees have wider boundaries than the analysis window, so is_SR==1 events
    can have m_llg in the sideband and vice versa). The mass window is the
    physical sideband definition and matches the CR sideband selection.
    (weight=1.0 for Data.)
    """
    base = DATA_SCORED_BASE.parent / f"{DATA_SCORED_BASE.name}{fold}"
    pieces = []
    for era in ALL_ERAS:
        p = base / "Data" / "Data" / f"{era}.root"
        if not p.exists():
            continue
        try:
            with uproot.open(str(p)) as f:
                t = f[TREE_NOM]
                arr = t.arrays([MASS, WEIGHT, SCORE, TREE_TAG, FOLDVAR], library="np")
        except Exception as exc:
            print(f"WARNING: cannot read {p}: {exc}", file=sys.stderr)
            continue
        # Pure mass-window sideband: matches the CR sideband definition.
        mask = np.isfinite(arr[MASS]) & in_sb_mass(arr[MASS])
        if not np.any(mask):
            continue
        pieces.append((
            np.asarray(arr[MASS][mask], np.float64),
            np.asarray(arr[WEIGHT][mask], np.float64),
            _stack_score(arr[SCORE])[mask],
            np.asarray(arr[TREE_TAG][mask], np.int32),
            np.asarray(arr[FOLDVAR][mask], np.int32),
        ))
    if not pieces:
        return empty_set()
    return EventSet(
        np.concatenate([x[0] for x in pieces]),
        np.concatenate([x[1] for x in pieces]),
        np.concatenate([x[2] for x in pieces]),
        np.concatenate([x[3] for x in pieces]),
        np.concatenate([x[4] for x in pieces]),
    )


def load_cr(fold):
    """CR Data (merged_CR_lowIDMVA) from fold `fold` scored dir.

    Returns (cr_sr, cr_sb) EventSets, split PURELY by mass:
      SR: m_llg in [120,130)
      SB: m_llg in [110,120) U [130,180]
    source_tree_tag is NOT used to restrict the SR/SB membership -- it marks
    which analysis tree (CR_lowIDMVA__<cat> vs CR_lowIDMVA_SB__<cat>) an event
    came from, but the original v3 CR trees have wider mass boundaries than the
    analysis window, so tag-based membership does not match the mass window.
    Using pure mass makes the CR SB definition identical to the nominal Data
    SB definition, so alpha = N_Data_SB / N_CR_SB uses one consistent
    sideband definition on both sides.
    Score is OOF ttH for this fold.
    """
    base = CR_SCORED_BASE.parent / f"{CR_SCORED_BASE.name}{fold}"
    masses, weights, scores, tags, folds = [], [], [], [], []
    for era in ALL_ERAS:
        p = base / "Data" / "Data" / f"{era}.root"
        if not p.exists():
            continue
        try:
            with uproot.open(str(p)) as f:
                t = f[TREE_CR]
                arr = t.arrays([MASS, WEIGHT, SCORE, TREE_TAG, FOLDVAR], library="np")
        except Exception as exc:
            print(f"WARNING: cannot read {p}: {exc}", file=sys.stderr)
            continue
        masses.append(np.asarray(arr[MASS], np.float64))
        weights.append(np.asarray(arr[WEIGHT], np.float64))
        scores.append(_stack_score(arr[SCORE]))
        tags.append(np.asarray(arr[TREE_TAG], np.int32))
        folds.append(np.asarray(arr[FOLDVAR], np.int32))
    if not masses:
        return empty_set(), empty_set()
    mass = np.concatenate(masses); weight = np.concatenate(weights)
    score = np.concatenate(scores); tag = np.concatenate(tags)
    foldvar = np.concatenate(folds)

    # Pure mass-window split (no tag restriction): identical definition to the
    # nominal Data sideband, so the transfer factor alpha is consistent.
    sr_mask = in_sr_mass(mass)
    sb_mask = in_sb_mass(mass)
    cr_sr = EventSet(mass[sr_mask], weight[sr_mask], score[sr_mask], tag[sr_mask], foldvar[sr_mask])
    cr_sb = EventSet(mass[sb_mask], weight[sb_mask], score[sb_mask], tag[sb_mask], foldvar[sb_mask])
    return cr_sr, cr_sb


def in_sr_mass(m):
    return (m > MASS_LO) & (m < MASS_HI)

def in_sb_mass(m):
    return ((m >= SB_LO1) & (m < SB_HI1)) | ((m >= SB_LO2) & (m <= SB_HI2))


# =============================================================================
# 4. Parity + category selection
# =============================================================================

def select_parity(es, parity):
    """Keep events with (photon_eta_fold % 4) == parity. OOF convention."""
    if es.n == 0:
        return es
    m = (es.foldvar % 4) == parity
    return EventSet(es.mass[m], es.weight[m], es.score[m], es.tag[m], es.foldvar[m])


def nom_cat_index(tag):
    """nominal source_tree_tag -> category index 0..4 (tag//2). -1 if not SR/SB cat."""
    return tag // 2

def cr_cat_index(tag):
    """CR source_tree_tag -> category index 0..4 (tag%5). Works for both SR(0-4) and SB(5-9)."""
    return tag % 5


# =============================================================================
# 5. 3-category evaluation with method-2 background
# =============================================================================

def evaluate_bins(sig, data_sb, cr_sr, cr_sb, c1, c2):
    """Evaluate 3 score-region bins (categories folded). Returns {region: result}.

    sig, data_sb, cr_sr, cr_sb are EventSets already parity-selected (val or test).

    The 5 analysis categories are FOLDED: all category events are merged into
    one population per score region. There is no per-category split, so the
    final Z is a quadrature combination of exactly 3 bins (high/medium/low).

    SR/SB are defined PURELY by mass everywhere, for consistency between S and B:
      SR = m_llg in (120,130)
      SB = m_llg in [110,120) U [130,180]
    is_SR / NOM_SR_TAGS are NOT used -- they mark which analysis tree (SR vs
    Sideband) an event came from, not the mass window; the original v3 trees have
    wider boundaries, so SB-tree events can have m_llg in the SR window and
    vice versa. Using pure mass on both S and B keeps the SR definition identical
    on the signal side and the background side.

    S  = ttH MC weighted yield in SR (pure mass), all cats merged, per score region.
    B  = (sum_N_Data_SB) * (sum_N_CR_SR) / (sum_N_CR_SB)
         i.e. one unified alpha over all cats, times the merged CR SR count.
    """
    results = {}
    # signal: ttH MC in the SR mass window (pure mass; no SR-tree restriction).
    sig_mass = sig.sig_mass_safe()
    sig_sr_mask = (sig.n > 0) & np.isfinite(sig_mass) & in_sr_mass(sig_mass)

    for region in ["low", "medium", "high"]:
        if region == "high":
            sb = (lambda s: s > c2)
        elif region == "medium":
            sb = (lambda s: (s > c1) & (s <= c2))
        else:
            sb = (lambda s: s <= c1)

        # --- S: ttH MC in SR + this region (all cats merged) ---
        sig_sel = sig_sr_mask & sb(sig.score)
        S = yield_info(sig.weight[sig_sel])

        # --- counts for B (all cats merged) ---
        n_data_sb = int(np.sum(sb(data_sb.score) & in_sb_mass(data_sb.mass))) if data_sb.n > 0 else 0
        n_cr_sr = int(np.sum(sb(cr_sr.score) & in_sr_mass(cr_sr.mass))) if cr_sr.n > 0 else 0
        n_cr_sb = int(np.sum(sb(cr_sb.score) & in_sb_mass(cr_sb.mass))) if cr_sb.n > 0 else 0

        # --- B + sigma_B ---
        B_val, sigma_b = sigma_b_transfer(n_data_sb, n_cr_sb, n_cr_sr)
        sigma_s = S["err"]

        valid = (B_val > 0.0) and (n_cr_sb >= MIN_N_CR_SB) and (S["sum"] >= 0.0)
        z = asimov_z(S["sum"], B_val) if valid else 0.0
        sigma_z, sigma_z_s, sigma_z_b = (
            z_uncertainty(S["sum"], B_val, sigma_s, sigma_b) if valid else (0.0, 0.0, 0.0)
        )

        results[region] = {
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
    """results is already {region: result} (categories folded). Pass through."""
    return results


# small helper so evaluate_bins can reference sig mass without assuming length
def _sig_mass_safe(self):
    return self.mass
EventSet.sig_mass_safe = _sig_mass_safe


# =============================================================================
# 6. Printing
# =============================================================================

def print_bins(title, results, c1, c2):
    flat = flatten(results)
    z, sigma_z = combined_z(flat)
    w = 110
    print("\n" + "=" * w)
    print(f"{title}: c1={c1:.6f}, c2={c2:.6f}")
    print("=" * w)
    hdr = (
        f"{'reg':7s}"
        f"{'S':>10}{'sigS':>9}"
        f"{'NdatSB':>8}{'NcrSB':>7}{'NcrSR':>7}"
        f"{'alpha':>8}{'B':>10}{'sigB':>9}"
        f"{'Z':>9}{'sigZ':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for region in ["high", "medium", "low"]:
        r = results[region]
        print(
            f"{region:7s}"
            f"{r['S']:>10.4f}{r['sigma_S']:>9.4f}"
            f"{r['N_data_sb']:>8d}{r['N_cr_sb']:>7d}{r['N_cr_sr']:>7d}"
            f"{r['alpha']:>8.3f}{r['B']:>10.3f}{r['sigma_B']:>9.3f}"
            f"{r['Z']:>9.4f}{r['sigma_Z']:>9.4f}"
        )
    print("-" * len(hdr))
    print(f"Combined Z = {z:.6f} ± {sigma_z:.6f}  (sqrt of sum of {sum(1 for r in flat.values() if r['valid'])} valid bins)")


# =============================================================================
# 7. Bayesian optimisation (val -> c1, c2)
# =============================================================================

def optimize_bayesian(sig_val, data_sb_val, cr_sr_val, cr_sb_val, label=""):
    if optuna is None:
        raise RuntimeError("optuna not available; pass --no-optuna with --c1/--c2")

    def objective(trial):
        c1 = trial.suggest_float("c1", C1_MIN, C1_MAX)
        c2_low = max(C2_MIN, c1 + MIN_GAP)
        if c2_low >= C2_MAX:
            return -1.0
        c2 = trial.suggest_float("c2", c2_low, C2_MAX)
        results = evaluate_bins(sig_val, data_sb_val, cr_sr_val, cr_sb_val, c1, c2)
        flat = flatten(results)
        z, _ = combined_z(flat)
        return z if z > 0.0 else -1.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, n_startup_trials=N_STARTUP_TRIALS
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nBayesian optimisation {label}: {N_TRIALS} trials ...")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    best = max(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value and t.value > 0),
        key=lambda t: t.value, default=None,
    )
    if best is None:
        raise RuntimeError(f"No valid trial for {label}; try --no-optuna --c1 --c2")
    c1 = float(best.params["c1"])
    c2 = float(best.params["c2"])
    return c1, c2


# =============================================================================
# 8. Main
# =============================================================================

def main():
    global N_TRIALS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-optuna", action="store_true",
                    help="skip optimisation; use --c1/--c2 for all folds")
    ap.add_argument("--c1", type=float, default=0.3)
    ap.add_argument("--c2", type=float, default=0.8)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    N_TRIALS = args.trials

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    for fold_idx in range(4):
        val_p  = VALID_PARITY[fold_idx]
        test_p = TEST_PARITY[fold_idx]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}  val parity={val_p}  test parity={test_p}")
        print(f"{'='*70}")

        # load all populations for this fold (score = this fold's OOF model)
        print("  Loading ttH MC signal ...")
        sig_all = load_nominal_mc(fold_idx, SIGNAL_GROUP[0], SIGNAL_GROUP[1])
        print(f"    N_sig={sig_all.n}")
        print("  Loading nominal Data SB ...")
        data_sb_all = load_data_sideband(fold_idx)
        print(f"    N_data_sb={data_sb_all.n}")
        print("  Loading CR Data (SR + SB) ...")
        cr_sr_all, cr_sb_all = load_cr(fold_idx)
        print(f"    N_cr_sr={cr_sr_all.n}  N_cr_sb={cr_sb_all.n}")

        # parity split (val / test)
        sig_val      = select_parity(sig_all, val_p)
        sig_test     = select_parity(sig_all, test_p)
        data_sb_val  = select_parity(data_sb_all, val_p)
        data_sb_test = select_parity(data_sb_all, test_p)
        cr_sr_val    = select_parity(cr_sr_all, val_p)
        cr_sr_test   = select_parity(cr_sr_all, test_p)
        cr_sb_val    = select_parity(cr_sb_all, val_p)
        cr_sb_test   = select_parity(cr_sb_all, test_p)

        print(f"  val : sig={sig_val.n} dataSB={data_sb_val.n} crSR={cr_sr_val.n} crSB={cr_sb_val.n}")
        print(f"  test: sig={sig_test.n} dataSB={data_sb_test.n} crSR={cr_sr_test.n} crSB={cr_sb_test.n}")

        # --- boundary optimisation on val ---
        if args.no_optuna:
            c1_best, c2_best = args.c1, args.c2
            print(f"  [skip optuna] using c1={c1_best} c2={c2_best}")
        else:
            c1_best, c2_best = optimize_bayesian(
                sig_val, data_sb_val, cr_sr_val, cr_sb_val,
                label=f"fold{fold_idx} val",
            )
        val_results = evaluate_bins(sig_val, data_sb_val, cr_sr_val, cr_sb_val, c1_best, c2_best)
        print_bins(f"FOLD {fold_idx} VALIDATION", val_results, c1_best, c2_best)
        z_val, _ = combined_z(flatten(val_results))

        # --- evaluate test with chosen boundaries ---
        test_results = evaluate_bins(sig_test, data_sb_test, cr_sr_test, cr_sb_test, c1_best, c2_best)
        print_bins(f"FOLD {fold_idx} TEST", test_results, c1_best, c2_best)
        z_test, sigma_z_test = combined_z(flatten(test_results))

        fold_results[fold_idx] = {
            "c1": c1_best, "c2": c2_best,
            "val_results": val_results, "test_results": test_results,
            "Z_val": z_val, "Z_test": z_test, "sigma_Z_test": sigma_z_test,
        }

    # --- combined across 4 folds ---
    print("\n" + "=" * 80)
    print("PER-FOLD SUMMARY")
    print("=" * 80)
    print(f"{'Fold':<6}{'c1':>10}{'c2':>10}{'Z_val':>12}{'Z_test':>14}{'sigZ_test':>12}")
    print("-" * 64)
    z_sq = 0.0
    for k, r in fold_results.items():
        print(f"{k:<6}{r['c1']:>10.6f}{r['c2']:>10.6f}"
              f"{r['Z_val']:>12.6f}{r['Z_test']:>14.6f}{r['sigma_Z_test']:>12.6f}")
        z_sq += r["Z_test"] ** 2
    z_final = float(np.sqrt(z_sq))

    sigma_sq = 0.0
    for r in fold_results.values():
        if z_final > 0.0 and r["Z_test"] > 0.0:
            sigma_sq += ((r["Z_test"] / z_final) * r["sigma_Z_test"]) ** 2
    sigma_final = float(np.sqrt(sigma_sq))

    print("\n" + "=" * 70)
    print("FINAL RESULT (method-2 CR transfer-factor background)")
    print("=" * 70)
    print(f"4-fold combined Z = {z_final:.6f} ± {sigma_final:.6f}")
    print(f"  = sqrt(" + " + ".join(f"{r['Z_test']:.4f}^2" for r in fold_results.values()) + ")")

    # save a short report
    rep = OUT_DIR / "final_z.txt"
    with open(rep, "w") as fh:
        fh.write("method-2 CR transfer-factor significance\n")
        fh.write(f"combined Z = {z_final:.6f} ± {sigma_final:.6f}\n\n")
        fh.write(f"{'fold':<6}{'c1':>10}{'c2':>10}{'Z_val':>12}{'Z_test':>14}{'sigZ':>12}\n")
        for k, r in fold_results.items():
            fh.write(f"{k:<6}{r['c1']:>10.6f}{r['c2']:>10.6f}"
                     f"{r['Z_val']:>12.6f}{r['Z_test']:>14.6f}{r['sigma_Z_test']:>12.6f}\n")
    print(f"\nReport: {rep}")


if __name__ == "__main__":
    main()
