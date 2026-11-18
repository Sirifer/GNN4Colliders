#!/usr/bin/env python3
"""
diag_cr_lowIDMVA_v3.py

Step-4 CR validation + method-2 background estimate for the v3 4-class setup.

Reads:
  - CR scored Data : root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold{0-3}/
                     tree = merged_CR_lowIDMVA, source_tree_tag 0-4 = CR SR,
                                                      5-9 = CR SB
  - nominal scored Data : root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold{0-3}/
                          tree = merged_SR_Sideband

Per the nominal convention, each event uses the score from the fold whose
test parity equals (photon_eta_fold % 4) -- i.e. parity k -> fold{k} scores
(the model that did NOT see parity k in training). So for every event we
pick one of the 4 fold directories by its photon_eta_fold % 4.

Method-2 background estimate (data-driven, per analysis category c and GNN
score region R):
    alpha(c,R) = N_Data_SB(c,R) / N_CR_SB(c,R)
    B_CR_SR(c,R) = alpha(c,R) * N_CR_SR(c,R)

This is identical to the method-1 transfer-factor form; method 2 frames the
same ratio as "scale CR data onto nominal sideband data, then read the
120-130 yield". Here we also report the pre-GNN (no score cut) version, which
is the pure mass-shape transfer factor the user described as step 2/3.

Mass windows (consistent with the established HZg convention):
    SR   : 120 <= m_llg < 130
    SB   : 110 <= m_llg < 120  OR  130 <= m_llg <= 180

Usage:
    python diag_cr_lowIDMVA_v3.py
    python diag_cr_lowIDMVA_v3.py --score-cut 0.5
    python diag_cr_lowIDMVA_v3.py --no-oof   # use fold0 scores for all (quick check)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import uproot

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")

CR_SCORED_BASE = REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold"
NOM_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data"

ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

CATS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]
# source_tree_tag -> category index, and SR (tag<5) vs SB (tag>=5)
# 0-4: CR_lowIDMVA__<cat>     (CR SR)
# 5-9: CR_lowIDMVA_SB__<cat>  (CR SB)
SR_TAG_OF = {c: i for i, c in enumerate(CATS)}          # 0..4
SB_TAG_OF = {c: 5 + i for i, c in enumerate(CATS)}      # 5..9

M_SR_LO, M_SR_HI = 120.0, 130.0
M_SB_LO1, M_SB_HI1 = 110.0, 120.0
M_SB_LO2, M_SB_HI2 = 130.0, 180.0

FOLDVAR = "photon_eta_fold"
SCORE_BRANCH = "ttH_gnn_score"   # 4-class -> vector<float> (4 entries)

OUT_DIR = REPO / "tools" / "results_cr_lowIDMVA_v3"


def in_sr(m):
    return (m >= M_SR_LO) & (m < M_SR_HI)

def in_sb(m):
    return ((m >= M_SB_LO1) & (m < M_SB_HI1)) | ((m >= M_SB_LO2) & (m <= M_SB_HI2))


def load_oof_scores(scored_base, tree_name, sub="Data/Data"):
    """Load m_llg, photon_eta_fold, source_tree_tag, and the OOF ttH score.

    For each event, pick the score from fold directory = (photon_eta_fold % 4).
    The 4-class score is a vector<float>; index 1 is the ttH class
    (label convention: bkg=0, ttH=1, VH=2, ggF+VBF=3 -- see config).
    Returns dict with arrays.
    """
    masses, folds, tags, scores = [], [], [], []
    found_folds = []
    for fold in range(4):
        d = scored_base / f"fold{fold}" / sub
        if not d.is_dir():
            continue
        found_folds.append(fold)

    if not found_folds:
        raise FileNotFoundError(f"no fold dirs under {scored_base}")

    # First gather per-fold arrays, then pick OOF per event.
    per_fold = {}
    for fold in found_folds:
        d = scored_base / f"fold{fold}" / sub
        fm, ff, ft, fs = [], [], [], []
        for era in ERAS:
            p = d / f"{era}.root"
            if not p.exists():
                continue
            with uproot.open(p) as f:
                t = f[tree_name]
                arr = t.arrays(
                    ["m_llg", FOLDVAR, "source_tree_tag", SCORE_BRANCH],
                    library="np",
                )
                fm.append(arr["m_llg"])
                ff.append(arr[FOLDVAR])
                ft.append(arr["source_tree_tag"])
                # vector<float> branch -> 2D array (N,4); take ttH index 1.
                # uproot reads std::vector<float> as an object dtype 1D array
                # (each entry a small ndarray), so stack it into (N,4) first.
                sc = arr[SCORE_BRANCH]
                if sc.dtype == object or sc.ndim == 1:
                    sc = np.stack([np.asarray(v, dtype=np.float32) for v in sc])
                fs.append(sc[:, 1])
        per_fold[fold] = (
            np.concatenate(fm), np.concatenate(ff),
            np.concatenate(ft), np.concatenate(fs),
        )

    # All folds score ALL events (verified on nominal), so arrays align across
    # folds. Stack scores (n_folds, N) and pick OOF row per event.
    base = per_fold[found_folds[0]]
    n = len(base[0])
    mass = base[0]
    foldvar = base[1]
    tag = base[2]
    score_stack = np.empty((len(found_folds), n), dtype=np.float32)
    for j, fold in enumerate(found_folds):
        if len(per_fold[fold][0]) != n:
            raise RuntimeError(
                f"fold{fold} has {len(per_fold[fold][0])} events, expected {n}; "
                "folds must score the same events"
            )
        score_stack[j] = per_fold[fold][3]

    parity = (foldvar % 4).astype(np.int64)
    # pick the fold directory matching this event's parity (OOF convention)
    pick_row = np.empty(n, dtype=np.int64)
    for j, fold in enumerate(found_folds):
        pick_row[parity == fold] = j
    oof_score = score_stack[pick_row, np.arange(n)]

    return {
        "m_llg": mass,
        "foldvar": foldvar,
        "source_tree_tag": tag,
        "score": oof_score,
        "parity": parity,
        "n_folds": len(found_folds),
    }


def cat_tag_mask(tags, cat, region):
    """region in {'sr','sb'}; returns mask for that cat's source_tree_tag."""
    t = SR_TAG_OF[cat] if region == "sr" else SB_TAG_OF[cat]
    return tags == t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-cut", type=float, default=None,
                    help="ttH score > cut for the GNN signal-enriched region; "
                         "if omitted, only pre-GNN (mass-shape) transfer is reported")
    ap.add_argument("--no-oof", action="store_true",
                    help="use fold0 scores for all events (quick check, not for final Z)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading nominal scored Data (OOF)...")
    nom = load_oof_scores(NOM_SCORED_BASE, "merged_SR_Sideband")
    print(f"  nominal: {len(nom['m_llg'])} events, {nom['n_folds']} folds")
    print("Loading CR scored Data (OOF)...")
    cr = load_oof_scores(CR_SCORED_BASE, "merged_CR_lowIDMVA")
    print(f"  CR     : {len(cr['m_llg'])} events, {cr['n_folds']} folds")
    print()

    # nominal SB: the nominal tree is merged_SR_Sideband; SR events have
    # source_tree_tag even (0,2,4,6,8), SB odd (1,3,5,7,9). For the method-2
    # denominator we need nominal DATA sideband events per category.
    nom_tag = nom["source_tree_tag"]
    nom_is_sb = (nom_tag % 2 == 1)
    nom_is_sr = (nom_tag % 2 == 0)
    # map nominal source_tree_tag -> category index: tag//2
    nom_cat = nom_tag // 2

    sep = "=" * 78
    print(sep)
    print("CR_lowIDMVA validation + method-2 background estimate (pre-GNN, mass shape)")
    print(sep)
    print(f"{'cat':9s} | {'CR_SR':>7s} {'CR_SB':>7s} | {'Nom_SB':>7s} | "
          f"{'alpha':>8s} {'B_CR_SR':>9s} {'B_err':>8s}")
    print("-" * 78)

    rows = []
    for cat in CATS:
        cr_sr = cat_tag_mask(cr["source_tree_tag"], cat, "sr") & in_sr(cr["m_llg"])
        cr_sb = cat_tag_mask(cr["source_tree_tag"], cat, "sb") & in_sb(cr["m_llg"])
        nom_sb = (nom_cat == CATS.index(cat)) & nom_is_sb & in_sb(nom["m_llg"])

        n_cr_sr = int(cr_sr.sum())
        n_cr_sb = int(cr_sb.sum())
        n_nom_sb = int(nom_sb.sum())

        if n_cr_sb > 0 and n_nom_sb >= 0:
            alpha = n_nom_sb / n_cr_sb
            b = alpha * n_cr_sr
            err = b * np.sqrt(1.0/max(n_nom_sb,1) + 1.0/n_cr_sb + 1.0/max(n_cr_sr,1)) if n_cr_sr > 0 else float("nan")
        else:
            alpha = b = err = float("nan")
        print(f"{cat:9s} | {n_cr_sr:7d} {n_cr_sb:7d} | {n_nom_sb:7d} | "
              f"{alpha:8.4f} {b:9.2f} {err:8.2f}")
        rows.append((cat, n_cr_sr, n_cr_sb, n_nom_sb, alpha, b, err))

    print("-" * 78)
    tot_cr_sr = sum(r[1] for r in rows)
    tot_cr_sb = sum(r[2] for r in rows)
    tot_nom_sb = sum(r[3] for r in rows)
    if tot_cr_sb > 0:
        a = tot_nom_sb / tot_cr_sb
        b = a * tot_cr_sr
        e = b * np.sqrt(1.0/max(tot_nom_sb,1) + 1.0/tot_cr_sb + 1.0/max(tot_cr_sr,1)) if tot_cr_sr > 0 else float("nan")
    else:
        a = b = e = float("nan")
    print(f"{'ALL':9s} | {tot_cr_sr:7d} {tot_cr_sb:7d} | {tot_nom_sb:7d} | "
          f"{a:8.4f} {b:9.2f} {e:8.2f}")
    print()

    if args.score_cut is not None:
        cut = args.score_cut
        print(sep)
        print(f"Method-2 with GNN score > {cut} (ttH class, OOF)")
        print(sep)
        print(f"{'cat':9s} | {'CR_SR':>7s} {'CR_SB':>7s} | {'Nom_SB':>7s} | "
              f"{'alpha':>8s} {'B_CR_SR':>9s} {'B_err':>8s}")
        print("-" * 78)
        rows2 = []
        for cat in CATS:
            cr_sr = (cat_tag_mask(cr["source_tree_tag"], cat, "sr")
                     & in_sr(cr["m_llg"]) & (cr["score"] > cut))
            cr_sb = (cat_tag_mask(cr["source_tree_tag"], cat, "sb")
                     & in_sb(cr["m_llg"]) & (cr["score"] > cut))
            nom_sb = ((nom_cat == CATS.index(cat)) & nom_is_sb
                      & in_sb(nom["m_llg"]) & (nom["score"] > cut))
            n1, n2, n3 = int(cr_sr.sum()), int(cr_sb.sum()), int(nom_sb.sum())
            if n2 > 0:
                al = n3 / n2; bb = al * n1
                ee = bb*np.sqrt(1.0/max(n3,1)+1.0/n2+1.0/max(n1,1)) if n1>0 else float("nan")
            else:
                al = bb = ee = float("nan")
            print(f"{cat:9s} | {n1:7d} {n2:7d} | {n3:7d} | {al:8.4f} {bb:9.2f} {ee:8.2f}")
            rows2.append((cat, n1, n2, n3, al, bb, ee))
        print()

    # score distribution sanity (finite, range)
    print(sep)
    print("Score sanity (OOF ttH class)")
    print(sep)
    for label, d in [("nominal", nom), ("CR", cr)]:
        s = d["score"]
        fin = np.isfinite(s)
        print(f"  {label:8s}: N={len(s)} finite={int(fin.sum())} "
              f"range=[{s[fin].min():.4f},{s[fin].max():.4f}] mean={s[fin].mean():.4f}")

    # save text report
    rep = OUT_DIR / "cr_lowIDMVA_v3_diag.txt"
    with open(rep, "w") as fh:
        fh.write("CR_lowIDMVA v3 method-2 diagnostic\n")
        fh.write(f"nominal events: {len(nom['m_llg'])}, CR events: {len(cr['m_llg'])}\n")
        fh.write(f"folds used (OOF): nominal={nom['n_folds']} CR={cr['n_folds']}\n\n")
        fh.write("pre-GNN (mass shape) transfer factor:\n")
        fh.write(f"{'cat':9s} CR_SR CR_SB Nom_SB alpha B err\n")
        for r in rows:
            fh.write(f"{r[0]:9s} {r[1]} {r[2]} {r[3]} {r[4]:.4f} {r[5]:.2f} {r[6]:.2f}\n")
    print(f"\nReport: {rep}")


if __name__ == "__main__":
    main()
