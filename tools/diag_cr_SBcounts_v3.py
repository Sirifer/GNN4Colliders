#!/usr/bin/env python3
"""
diag_cr_SBcounts_v3.py

Per-fold, per (cat x region) raw counts for method-2, with B recomputed from
the three counts, plus S event count and weighted yield.

For each fold (TEST parity) and each (cat, region):
  N_Data_SB  = nominal Data events, mass in SB [110,120) U [130,180], this cat+region
  N_CR_SB    = CR Data events,        mass in SB, this cat+region
  N_CR_SR    = CR Data events,        mass in SR (120,130), this cat+region
  alpha      = N_Data_SB / N_CR_SB   (nan if N_CR_SB==0)
  B          = alpha * N_CR_SR        (0 if alpha undefined or N_CR_SR==0)

  S_count    = ttH MC events in SR, this cat+region (unweighted count)
  S_yield    = sum(weight_lumi) of those events       (weighted yield)

Selection matches significance_4class_v3_cr.py exactly (pure mass, OOF fold,
cat: nominal tag//2 / CR tag%5, score regions with fixed c1=0.3 c2=0.8).

Usage:
    python diag_cr_SBcounts_v3.py
"""
from pathlib import Path
import importlib.util
import numpy as np

TOOLS = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/tools")
_spec = importlib.util.spec_from_file_location("zmod", TOOLS / "significance_4class_v3_cr.py")
z = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z)

CATS = z.CATS
TEST_PARITY = z.TEST_PARITY
C1, C2 = 0.3, 0.8

def in_sr(m): return (m > z.MASS_LO) & (m < z.MASS_HI)
def in_sb(m): return ((m >= z.SB_LO1) & (m < z.SB_HI1)) | ((m >= z.SB_LO2) & (m <= z.SB_HI2))
def reg_mask(s, reg):
    if reg == "high":   return s > C2
    if reg == "medium": return (s > C1) & (s <= C2)
    return s <= C1
REGIONS = ["high", "medium", "low"]

def cnt(es, ci, reg, mass_in, cidx):
    if es.n == 0: return 0
    return int(np.sum((cidx(es.tag)==ci) & mass_in(es.mass) & reg_mask(es.score, reg)))

def s_count_yield(sig, ci, reg):
    """ttH MC in SR, this cat+region: (unweighted count, weighted yield)."""
    if sig.n == 0: return 0, 0.0
    sig_mass = sig.mass
    sr = (np.isfinite(sig_mass) & in_sr(sig_mass) &
          (z.nom_cat_index(sig.tag) == ci) & reg_mask(sig.score, reg))
    w = sig.weight[sr]
    return int(w.size), float(np.sum(w))

def main():
    sep = "=" * 118
    print(sep)
    print(f"Per-fold raw counts + recomputed B + S(count,yield)  [merged alpha, pure mass, OOF, c1={C1} c2={C2}]")
    print(f"    alpha = N_Data_SB / N_CR_SB ;  B = alpha * N_CR_SR ;  S_count = #ttH MC in SR ;  S_yield = sum(weight_lumi)")
    print(sep)

    grand = {"B":0.0, "S_yield":0.0, "S_count":0,
             "N_Data_SB":0, "N_CR_SB":0, "N_CR_SR":0}
    for fold in range(4):
        tp = TEST_PARITY[fold]
        sig  = z.select_parity(z.load_nominal_mc(fold, z.SIGNAL_GROUP[0], z.SIGNAL_GROUP[1]), tp)
        data_sb = z.select_parity(z.load_data_sideband(fold), tp)
        cr_sr, cr_sb = z.load_cr(fold)
        cr_sr = z.select_parity(cr_sr, tp)
        cr_sb = z.select_parity(cr_sb, tp)

        hdr = (f"{'cat':9s} {'reg':7s} "
               f"{'NdatSB':>7} {'NcrSB':>6} {'NcrSR':>6} {'alpha':>7} {'B':>9}   "
               f"{'S_cnt':>7} {'S_yield':>9}")
        print(f"\n--- FOLD {fold}  (test parity={tp}) ---")
        print(hdr); print("-"*len(hdr))
        fB = fSy = 0.0; fSc = fND = fNC = fNR = 0
        for ci, cat in enumerate(CATS):
            for reg in REGIONS:
                nd  = cnt(data_sb, ci, reg, in_sb, z.nom_cat_index)
                ncs = cnt(cr_sb,   ci, reg, in_sb, z.cr_cat_index)
                ncr = cnt(cr_sr,   ci, reg, in_sr, z.cr_cat_index)
                alpha = nd/ncs if ncs>0 else float("nan")
                B = alpha*ncr if (ncs>0 and ncr>0) else 0.0
                sc, sy = s_count_yield(sig, ci, reg)
                fB+=B; fSy+=sy; fSc+=sc; fND+=nd; fNC+=ncs; fNR+=ncr
                a_str = f"{alpha:>7.3f}" if alpha==alpha else f"{'nan':>7s}"
                print(f"{cat:9s} {reg:7s} {nd:>7d} {ncs:>6d} {ncr:>6d} {a_str} {B:>9.2f}   {sc:>7d} {sy:>9.4f}")
        print("-"*len(hdr))
        print(f"{'FOLD '+str(fold):17s} {fND:>7d} {fNC:>6d} {fNR:>6d} {'':>7} {fB:>9.2f}   {fSc:>7d} {fSy:>9.4f}")
        grand["B"]+=fB; grand["S_yield"]+=fSy; grand["S_count"]+=fSc
        grand["N_Data_SB"]+=fND; grand["N_CR_SB"]+=fNC; grand["N_CR_SR"]+=fNR

    print("\n"+sep)
    print("4-FOLD TOTALS (TEST parity)")
    print(sep)
    print(f"  N_Data_SB total : {grand['N_Data_SB']}")
    print(f"  N_CR_SB   total : {grand['N_CR_SB']}")
    print(f"  N_CR_SR   total : {grand['N_CR_SR']}")
    print(f"  alpha overall   : {grand['N_Data_SB']/grand['N_CR_SB']:.4f}")
    print(f"  B  total (sum)  : {grand['B']:.3f}")
    print(f"  S  count total  : {grand['S_count']}")
    print(f"  S  yield total  : {grand['S_yield']:.4f}")

if __name__ == "__main__":
    main()
