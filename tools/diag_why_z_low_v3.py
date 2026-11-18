#!/usr/bin/env python3
"""
diag_why_z_low_v3.py

Why is the final Z so low (~0.5)? Break down, per score region (4 folds merged on TEST):
  - S_yield, B (sum over 5 cats), S/B, and the Z^2 contribution from that region.
  - CR SR vs CR SB score distributions: does CR have enough events in the high region,
    and is the CR score shape the bottleneck?

Also prints the per-region Z^2 share so we can see which region is actually limiting.

Selection matches significance_4class_v3_cr.py (pure mass, OOF, c1=0.3 c2=0.8).
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

def asimov_z(s, b):
    if s<=0 or b<=0: return 0.0
    x = 2*((s+b)*np.log1p(s/b)-s)
    return float(np.sqrt(x)) if x>0 else 0.0

def main():
    # merge all 4 folds TEST parity
    sigs=[]; datas=[]; crsrs=[]; crsbs=[]
    for fold in range(4):
        tp=TEST_PARITY[fold]
        sigs.append(z.select_parity(z.load_nominal_mc(fold, z.SIGNAL_GROUP[0], z.SIGNAL_GROUP[1]), tp))
        datas.append(z.select_parity(z.load_data_sideband(fold), tp))
        cr_sr, cr_sb = z.load_cr(fold)
        crsrs.append(z.select_parity(cr_sr, tp))
        crsbs.append(z.select_parity(cr_sb, tp))

    def cat_idx(es, fn):
        return fn(es.tag) if es.n else np.array([], np.int32)

    print("="*92)
    print(f"Per score-region breakdown (4 folds TEST merged, c1={C1} c2={C2})")
    print("="*92)
    hdr=f"{'region':7s} {'S_yield':>9} {'B':>10} {'S/B':>8} {'Z(region)':>10} {'Z^2 share':>10}"
    print(hdr); print("-"*len(hdr))

    z2_total=0.0; region_data={}
    for reg in REGIONS:
        S=0.0; B=0.0; Sc=0
        # also accumulate per-region CR SR/SB counts and score arrays for shape check
        cr_sr_scores=[]; cr_sb_scores=[]
        for i in range(4):
            sig=sigs[i]; data=datas[i]; crsr=crsrs[i]; crsb=crsbs[i]
            # S: ttH MC in SR
            if sig.n:
                m=sig.mass; sr=np.isfinite(m)&in_sr(m)
                for ci in range(len(CATS)):
                    sel = sr & (cat_idx(sig,z.nom_cat_index)==ci) & reg_mask(sig.score, reg)
                    S += float(np.sum(sig.weight[sel])); Sc += int(sel.sum())
            # B: per cat alpha*N_CR_SR
            if data.n: dcat=cat_idx(data,z.nom_cat_index)
            else: dcat=np.array([],np.int32)
            if crsr.n: csc=cat_idx(crsr,z.cr_cat_index)
            else: csc=np.array([],np.int32)
            if crsb.n: cbc=cat_idx(crsb,z.cr_cat_index)
            else: cbc=np.array([],np.int32)
            for ci in range(len(CATS)):
                nd=int(np.sum((dcat==ci)&in_sb(data.mass)&reg_mask(data.score,reg))) if data.n else 0
                ncs=int(np.sum((cbc==ci)&in_sb(crsb.mass)&reg_mask(crsb.score,reg))) if crsb.n else 0
                ncr=int(np.sum((csc==ci)&in_sr(crsr.mass)&reg_mask(crsr.score,reg))) if crsr.n else 0
                if ncs>0 and ncr>0:
                    B += (nd/ncs)*ncr
            # collect CR scores in SR/SB for this region (all cats)
            if crsr.n:
                m=crsr.mass; sr=np.isfinite(m)&in_sr(m)&reg_mask(crsr.score,reg)
                cr_sr_scores.append(crsr.score[sr])
            if crsb.n:
                m=crsb.mass; sb=np.isfinite(m)&in_sb(m)&reg_mask(crsb.score,reg)
                cr_sb_scores.append(crsb.score[sb])
        Zr=asimov_z(S,B)
        z2_total += Zr**2
        region_data[reg]=dict(S=S,B=B,Sc=Sc,Z=Zr,
                              cr_sr=np.concatenate(cr_sr_scores) if cr_sr_scores else np.array([]),
                              cr_sb=np.concatenate(cr_sb_scores) if cr_sb_scores else np.array([]))
        print(f"{reg:7s} {S:>9.4f} {B:>10.2f} {S/B if B>0 else 0:>8.5f} {Zr:>10.4f} {Zr**2:>10.4f}")
    print("-"*len(hdr))
    print(f"{'TOTAL':7s} {'':>9} {'':>10} {'':>8} {np.sqrt(z2_total):>10.4f} {z2_total:>10.4f}")
    print(f"  (sum of Z^2 over 3 regions, sqrt -> combined Z)")

    # ---- CR score shape: is the high region starved? ----
    print("\n"+"="*92)
    print("CR score distribution (4 folds TEST merged, all cats, SR mass vs SB mass)")
    print("="*92)
    # gather full CR SR / SB score (no region cut) for shape
    cr_sr_all=[]; cr_sb_all=[]
    for i in range(4):
        crsr=crsrs[i]; crsb=crsbs[i]
        if crsr.n:
            m=crsr.mass; sr=np.isfinite(m)&in_sr(m)
            cr_sr_all.append(crsr.score[sr])
        if crsb.n:
            m=crsb.mass; sb=np.isfinite(m)&in_sb(m)
            cr_sb_all.append(crsb.score[sb])
    cr_sr_s=np.concatenate(cr_sr_all) if cr_sr_all else np.array([])
    cr_sb_s=np.concatenate(cr_sb_all) if cr_sb_all else np.array([])
    print(f"  CR SR (mass 120-130): N={len(cr_sr_s)}")
    if len(cr_sr_s):
        for lo,hi,name in [(C2,1.01,"high"),(C1,C2,"medium"),(0.0,C1,"low")]:
            n=int(np.sum((cr_sr_s>lo)&(cr_sr_s<=hi)))
            print(f"    {name:7s} (score {lo:.2f}-{hi:.2f}): {n:6d}  ({n/len(cr_sr_s)*100:5.1f}%)")
    print(f"  CR SB (mass SB):      N={len(cr_sb_s)}")
    if len(cr_sb_s):
        for lo,hi,name in [(C2,1.01,"high"),(C1,C2,"medium"),(0.0,C1,"low")]:
            n=int(np.sum((cr_sb_s>lo)&(cr_sb_s<=hi)))
            print(f"    {name:7s} (score {lo:.2f}-{hi:.2f}): {n:6d}  ({n/len(cr_sb_s)*100:5.1f}%)")

    # ---- signal score shape for comparison ----
    print("\n"+"="*92)
    print("ttH MC (signal) score distribution in SR (4 folds TEST merged, all cats)")
    print("="*92)
    sig_scores=[]
    for i in range(4):
        sig=sigs[i]
        if sig.n:
            m=sig.mass; sr=np.isfinite(m)&in_sr(m)
            sig_scores.append(sig.score[sr])
    sig_s=np.concatenate(sig_scores) if sig_scores else np.array([])
    print(f"  Signal SR: N={len(sig_s)}")
    if len(sig_s):
        for lo,hi,name in [(C2,1.01,"high"),(C1,C2,"medium"),(0.0,C1,"low")]:
            n=int(np.sum((sig_s>lo)&(sig_s<=hi)))
            print(f"    {name:7s} (score {lo:.2f}-{hi:.2f}): {n:6d}  ({n/len(sig_s)*100:5.1f}%)")

    # ---- the real bottleneck: S/B per region ----
    print("\n"+"="*92)
    print("Bottleneck analysis")
    print("="*92)
    for reg in REGIONS:
        d=region_data[reg]
        print(f"  {reg:7s}: S={d['S']:.4f}  B={d['B']:.2f}  S/B={d['S']/d['B'] if d['B']>0 else 0:.5f}  "
              f"Z={d['Z']:.4f}  (Z^2={d['Z']**2:.4f}, share={d['Z']**2/z2_total*100 if z2_total>0 else 0:.1f}%)")
    print(f"\n  Combined Z = sqrt({z2_total:.4f}) = {np.sqrt(z2_total):.4f}")
    print(f"  -> Z is dominated by the region with the best S/B, NOT the most events.")
    print(f"  -> low region has huge B (~{region_data['low']['B']:.0f}) but tiny S/B -> negligible Z.")
    print(f"  -> high region has small B but decent S/B -> provides most of Z^2.")

if __name__ == "__main__":
    main()
