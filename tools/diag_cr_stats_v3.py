#!/usr/bin/env python3
"""
diag_cr_stats_v3.py

CR statistics diagnostic for method-2 (v3, 4-class, 4-fold OOF).

Goal (per the user's review of the 2026-11-03 run): figure out, among the
4 fold x 5 category x 3 score-region = 60 bins, which have enough CR
statistics and which are killed by zero/low counts. This decides whether the
current B=0-on-zero-count handling and the Optuna boundary search are trustworthy.

Selection matches significance_4class_v3_cr.py EXACTLY (it imports that module's
loaders), so the counts here are the same numbers that go into the Z calc:
  - SR / SB split PURELY by mass (no is_SR, no source_tree_tag restriction)
      SR = m_llg in (120, 130)
      SB = m_llg in [110,120) U [130,180]
  - category: nominal tag//2, CR tag%5
  - score region with fixed c1=0.3, c2=0.8 (same as the merged-table review)

Also computes the two-sideband transfer factor split (problem 5):
  alpha_lowSB  = N(m_llg in 110-120) on nominal / CR
  alpha_highSB = N(m_llg in 130-180) on nominal / CR
per category (all folds merged, for statistics), to check whether a single
unified alpha is defensible.

Usage:
    python diag_cr_stats_v3.py
"""
import sys
from pathlib import Path
import importlib.util
import numpy as np

TOOLS = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/tools")
# load the Z module without importing as a package
_spec = importlib.util.spec_from_file_location("zmod", TOOLS / "significance_4class_v3_cr.py")
z = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z)

CATS = z.CATS
ALL_ERAS = z.ALL_ERAS
TEST_PARITY = z.TEST_PARITY
VALID_PARITY = z.VALID_PARITY

# mass-window helpers (same constants as the Z script)
MASS_LO, MASS_HI = z.MASS_LO, z.MASS_HI
SB_LO1, SB_HI1 = z.SB_LO1, z.SB_HI1
SB_LO2, SB_HI2 = z.SB_LO2, z.SB_HI2
def in_sr(m): return (m > MASS_LO) & (m < MASS_HI)
def in_sb(m): return ((m >= SB_LO1) & (m < SB_HI1)) | ((m >= SB_LO2) & (m <= SB_HI2))
def in_sb_lo(m): return (m >= SB_LO1) & (m < SB_HI1)     # 110-120
def in_sb_hi(m): return (m >= SB_LO2) & (m <= SB_HI2)    # 130-180

# fixed score thresholds (matches the merged review table)
C1, C2 = 0.3, 0.8
def region_mask(score, reg):
    if reg == "high":   return score > C2
    if reg == "medium": return (score > C1) & (score <= C2)
    return score <= C1   # low
REGIONS = ["high", "medium", "low"]


def count_in(es, cat_idx, reg, mass_in, cat_index_fn):
    """count events of es in (cat, region) with a mass-window mask."""
    if es.n == 0:
        return 0
    cm = (cat_index_fn(es.tag) == cat_idx) & mass_in(es.mass) & region_mask(es.score, reg)
    return int(np.sum(cm))


def main():
    print("=" * 92)
    print(f"CR statistics diagnostic  (selection = pure mass, OOF, c1={C1} c2={C2})")
    print("=" * 92)

    # ----------------------------------------------------------------------
    # Part A: per (fold, cat, region) counts on the TEST parity
    # ----------------------------------------------------------------------
    header = (f"{'fold':>4} {'cat':9s} {'reg':7s} "
              f"{'NdatSB':>7} {'NcrSB':>6} {'NcrSR':>6} "
              f"{'alpha':>7} {'B':>8} {'sigB/B':>7} {'flag':>10}")
    print("\n" + header)
    print("-" * len(header))

    n_bins = 0
    n_zero_crsb = 0      # N_CR_SB == 0  (alpha undefined -> B forced 0)
    n_zero_crsr = 0      # N_CR_SR == 0  (B -> 0 by current code)
    n_zero_both = 0
    n_low_crsb = 0       # N_CR_SB in 1..4 (alpha unstable)
    n_low_crsr = 0       # N_CR_SR in 1..4
    n_ok = 0             # N_CR_SB>=5 and N_CR_SR>=5
    per_fold_valid = {f: 0 for f in range(4)}

    for fold in range(4):
        sig_all = z.load_nominal_mc(fold, z.SIGNAL_GROUP[0], z.SIGNAL_GROUP[1])
        data_sb_all = z.load_data_sideband(fold)
        cr_sr_all, cr_sb_all = z.load_cr(fold)
        tp = TEST_PARITY[fold]
        sig_test = z.select_parity(sig_all, tp)
        data_sb = z.select_parity(data_sb_all, tp)
        cr_sr = z.select_parity(cr_sr_all, tp)
        cr_sb = z.select_parity(cr_sb_all, tp)

        for ci, cat in enumerate(CATS):
            for reg in REGIONS:
                n_data_sb = count_in(data_sb, ci, reg, in_sb, z.nom_cat_index)
                n_cr_sb   = count_in(cr_sb,   ci, reg, in_sb, z.cr_cat_index)
                n_cr_sr   = count_in(cr_sr,   ci, reg, in_sr, z.cr_cat_index)
                n_bins += 1
                if n_cr_sb == 0: n_zero_crsb += 1
                if n_cr_sr == 0: n_zero_crsr += 1
                if n_cr_sb == 0 and n_cr_sr == 0: n_zero_both += 1
                if 1 <= n_cr_sb <= 4: n_low_crsb += 1
                if 1 <= n_cr_sr <= 4: n_low_crsr += 1
                if n_cr_sb >= 5 and n_cr_sr >= 5:
                    n_ok += 1
                    per_fold_valid[fold] += 1

                if n_cr_sb > 0:
                    alpha = n_data_sb / n_cr_sb
                    B = alpha * n_cr_sr
                    if B > 0:
                        rel = np.sqrt(1.0/max(n_data_sb,1) + 1.0/n_cr_sb + 1.0/max(n_cr_sr,1))
                    else:
                        rel = float("nan")
                else:
                    alpha = float("nan"); B = 0.0; rel = float("nan")

                # flag
                flags = []
                if n_cr_sb == 0: flags.append("SB=0")
                elif n_cr_sb < 5: flags.append("SBlo")
                if n_cr_sr == 0: flags.append("SR=0")
                elif n_cr_sr < 5: flags.append("SRlo")
                flag = ",".join(flags) if flags else "ok"

                print(f"{fold:>4} {cat:9s} {reg:7s} "
                      f"{n_data_sb:>7d} {n_cr_sb:>6d} {n_cr_sr:>6d} "
                      f"{alpha:>7.3f} {B:>8.2f} {(rel if rel==rel else 0):>7.3f} {flag:>10}")
        print("-" * len(header))

    print("\n" + "=" * 92)
    print("SUMMARY over 4 fold x 5 cat x 3 region = 60 TEST bins")
    print("=" * 92)
    print(f"  total bins                : {n_bins}")
    print(f"  N_CR_SB == 0 (alpha undef): {n_zero_crsb}   -> B forced to 0")
    print(f"  N_CR_SR == 0 (B -> 0)     : {n_zero_crsr}")
    print(f"  both zero                 : {n_zero_both}")
    print(f"  N_CR_SB in 1..4 (unstable): {n_low_crsb}")
    print(f"  N_CR_SR in 1..4 (unstable): {n_low_crsr}")
    print(f"  robust (SB>=5 and SR>=5)  : {n_ok}")
    print(f"  per-fold robust bins      : " + "  ".join(f"f{f}={per_fold_valid[f]}" for f in range(4)))

    # ----------------------------------------------------------------------
    # Part B: two-sideband alpha split (problem 5), all folds merged per cat
    # ----------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("Two-sideband transfer factor (problem 5): alpha_low(110-120) vs alpha_high(130-180)")
    print("  all 4 folds merged per category; counts are pure-mass, no score cut")
    print("=" * 92)
    # merge all folds (no parity) for max stats
    def merge(loader, *args):
        es_list = [loader(f, *args) for f in range(4)]
        return es_list
    data_all = [z.load_data_sideband(f) for f in range(4)]
    cr_sr_all_list = [z.load_cr(f) for f in range(4)]

    hdr = (f"{'cat':9s} "
           f"{'NDlo':>6} {'NClo':>5} {'a_lo':>7}  "
           f"{'NDhi':>6} {'NChi':>5} {'a_hi':>7}  "
           f"{'ratio hi/lo':>11}  {'verdict':>10}")
    print(hdr)
    print("-" * len(hdr))
    for ci, cat in enumerate(CATS):
        nd_lo = nc_lo = nd_hi = nc_hi = 0
        for f in range(4):
            d = data_all[f]; cr = cr_sr_all_list[f]
            if d.n:
                dcat = z.nom_cat_index(d.tag)
                nd_lo += int(np.sum((dcat==ci) & in_sb_lo(d.mass)))
                nd_hi += int(np.sum((dcat==ci) & in_sb_hi(d.mass)))
            c = cr[1]  # cr_sb
            if c.n:
                ccat = z.cr_cat_index(c.tag)
                nc_lo += int(np.sum((ccat==ci) & in_sb_lo(c.mass)))
                nc_hi += int(np.sum((ccat==ci) & in_sb_hi(c.mass)))
        a_lo = nd_lo/nc_lo if nc_lo>0 else float("nan")
        a_hi = nd_hi/nc_hi if nc_hi>0 else float("nan")
        ratio = a_hi/a_lo if (a_lo==a_lo and a_lo>0) else float("nan")
        verdict = "consistent" if (ratio==ratio and 0.8<ratio<1.2) else ("check" if ratio==ratio else "no-stats")
        print(f"{cat:9s} {nd_lo:>6d} {nc_lo:>5d} {a_lo:>7.3f}  "
              f"{nd_hi:>6d} {nc_hi:>5d} {a_hi:>7.3f}  {ratio:>11.3f}  {verdict:>10}")
    print("-" * len(hdr))
    print("  ratio hi/lo near 1.0 -> single unified alpha is defensible")


if __name__ == "__main__":
    main()
