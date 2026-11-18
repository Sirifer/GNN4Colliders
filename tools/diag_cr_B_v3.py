#!/usr/bin/env python3
"""
diag_cr_B_v3.py

Per-fold (5 cat x 3 score region) background B for method-2, with the SAME
merged-alpha definition as significance_4class_v3_cr.py:

    alpha = N_Data_SB / N_CR_SB          (SB = [110,120) U [130,180], merged)
    B     = alpha * N_CR_SR              (SR = (120,130))

Selection matches the Z script exactly (pure mass, OOF fold, fixed c1=0.3 c2=0.8).
Counts are on each fold's TEST parity (the parity used for the final Z).
Reports per-fold totals and the 4-fold combined B.

Usage:
    python diag_cr_B_v3.py
"""
import sys
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

def main():
    print("=" * 104)
    print(f"Method-2 background B per (fold, cat, region)  [merged alpha, pure mass, OOF, c1={C1} c2={C2}]")
    print(f"    alpha = N_Data_SB / N_CR_SB ;  B = alpha * N_CR_SR   (TEST parity of each fold)")
    print("=" * 104)
    hdr = (f"{'fold':>4} {'cat':9s} {'reg':7s} "
           f"{'NdatSB':>7} {'NcrSB':>6} {'NcrSR':>6} {'alpha':>7} {'B':>9}")
    print(hdr); print("-"*len(hdr))

    grand_B = 0.0
    grand_B_by_cat = {c: 0.0 for c in CATS}
    grand_B_by_reg = {r: 0.0 for r in REGIONS}
    fold_totals = {}

    for fold in range(4):
        data_sb = z.select_parity(z.load_data_sideband(fold), TEST_PARITY[fold])
        cr_sr, cr_sb = z.load_cr(fold)
        cr_sr = z.select_parity(cr_sr, TEST_PARITY[fold])
        cr_sb = z.select_parity(cr_sb, TEST_PARITY[fold])

        fold_B = 0.0
        for ci, cat in enumerate(CATS):
            for reg in REGIONS:
                nd = cnt(data_sb, ci, reg, in_sb, z.nom_cat_index)
                ncs = cnt(cr_sb,   ci, reg, in_sb, z.cr_cat_index)
                ncr = cnt(cr_sr,   ci, reg, in_sr, z.cr_cat_index)
                alpha = nd/ncs if ncs>0 else float("nan")
                B = alpha*ncr if (ncs>0 and ncr>0) else 0.0
                fold_B += B
                grand_B_by_cat[cat] += B
                grand_B_by_reg[reg] += B
                print(f"{fold:>4} {cat:9s} {reg:7s} {nd:>7d} {ncs:>6d} {ncr:>6d} "
                      f"{alpha:>7.3f} {B:>9.2f}")
        print("-"*len(hdr))
        print(f"{fold:>4} {'FOLD TOTAL':24s} {'':>20} {fold_B:>9.2f}")
        print("-"*len(hdr))
        fold_totals[fold] = fold_B
        grand_B += fold_B

    print("\n" + "=" * 104)
    print("4-FOLD TOTALS (TEST parity)")
    print("=" * 104)
    print(f"  {'fold':>6}{'B_total':>14}")
    for f in range(4):
        print(f"  {f:>6}{fold_totals[f]:>14.3f}")
    print(f"  {'sum':>6}{grand_B:>14.3f}   (sum of per-fold B)")

    print(f"\n  B by category (4 folds):")
    for c in CATS:
        print(f"    {c:9s}: {grand_B_by_cat[c]:>12.3f}")
    print(f"\n  B by score region (4 folds):")
    for r in REGIONS:
        print(f"    {r:7s}: {grand_B_by_reg[r]:>12.3f}")

if __name__ == "__main__":
    main()
