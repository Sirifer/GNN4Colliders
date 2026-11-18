#!/usr/bin/env python3
"""
Compute Asimov significance Z = sqrt(2*[(S+B)*ln(1+S/B) - S])
for ttH signal vs all backgrounds as a function of GNN score threshold.

Signal:     root_p2root_v2_scored/Signal/ttH/  (Run 2 + Run 3)
Background: root_p2root_v2_scored/Signal/{ggF,VBF,VH}/ + root_p2root_v2_scored/Bkg_MC/  (Run 2 + Run 3)
Selection:  is_SR == 1, 115 < m_llg < 135 GeV
Weight:     weight_lumi
Folds:      all events by default (use --test-fold-only for event % 4 == 0)

Note on corrections vs naive estimate from test-fold-only, 120-130 GeV window:
  mass window 120->115-135: effectively doubles background window, affects Z
  test fold -> all folds:   4x events -> ~2x significance (if GNN scores are cross-val unbiased)
  Run2 added (138/fb):      sqrt((283+138)/283) ~ 1.22x
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALL_YEARS = {
    "2016postVFP","2016preVFP","2017","2018",          # Run 2
    "2022preEE","2022postEE","2023preBPix","2023postBPix","2024","2025"  # Run 3
}
MASS_LO = 120.0
MASS_HI = 130.0
BRANCHES = ["ttH_gnn_score", "weight_lumi", "m_llg", "is_SR", "event"]


def load_files(root_dir, test_fold_only=False):
    scores, weights = [], []
    files = sorted(Path(root_dir).rglob("*.root"))
    files = [f for f in files if f.stem in ALL_YEARS]
    if not files:
        print(f"  WARNING: no ROOT files in {root_dir}", file=sys.stderr)
        return np.array([]), np.array([])
    for path in files:
        try:
            with uproot.open(str(path)) as f:
                tree = f["merged_SR_Sideband"]
                arrays = tree.arrays(BRANCHES, library="np")
        except Exception as e:
            print(f"  WARNING: skipping {path}: {e}", file=sys.stderr)
            continue
        mask = (arrays["is_SR"] == 1) & (arrays["m_llg"] > MASS_LO) & (arrays["m_llg"] < MASS_HI)
        if test_fold_only:
            mask &= (arrays["event"] % 4 == 0)
        scores.append(arrays["ttH_gnn_score"][mask])
        weights.append(arrays["weight_lumi"][mask])
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(weights)


def asimov_significance(s, b):
    if b <= 0:
        return 0.0
    if s <= 0:
        return 0.0
    return np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s))


def three_cat_significance(sig_scores, sig_weights, bkg_scores, bkg_weights,
                           sig_mask, bkg_mask, t_low, t_high):
    """Z_comb = sqrt(Z_high^2 + Z_mid^2 + Z_low^2) for three GNN score categories."""
    cats = [
        (sig_scores > t_high,                                         bkg_scores > t_high),
        ((sig_scores > t_low) & (sig_scores <= t_high),               (bkg_scores > t_low) & (bkg_scores <= t_high)),
        (sig_scores <= t_low,                                          bkg_scores <= t_low),
    ]
    zs = []
    sb = []
    for sel_s, sel_b in cats:
        S = sig_weights[sig_mask & sel_s].sum()
        B = bkg_weights[bkg_mask & sel_b].sum()
        zs.append(asimov_significance(S, B))
        sb.append((S, B))
    z_comb = np.sqrt(sum(z**2 for z in zs))
    return z_comb, zs, sb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir", default="/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored_run2run3")
    parser.add_argument("--output-dir", default="/scratchfs2/cms/huangsuyun/GNN4Colliders/eval/hzg/ttH_vs_other_run2run3_v2")
    parser.add_argument("--test-fold-only", action="store_true",
                        help="Use only test fold (event %% 4 == 0). Default: all folds.")
    args = parser.parse_args()

    scored = Path(args.scored_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load signal (ttH, Run 2 + Run 3) ---
    fold_label = "test fold (event%4==0)" if args.test_fold_only else "all folds"
    print(f"Loading signal: Signal/ttH ({fold_label}) ...", flush=True)
    sig_scores, sig_weights = load_files(scored / "Signal" / "ttH", test_fold_only=args.test_fold_only)
    print(f"  {len(sig_scores)} events, sum(w) = {sig_weights.sum():.4g}", flush=True)

    # --- load backgrounds: other signal processes + all Bkg_MC ---
    bkg_scores_list, bkg_weights_list = [], []
    for bkg_dir in [scored / "Signal" / "ggF",
                    scored / "Signal" / "VBF",
                    scored / "Signal" / "VH",
                    scored / "Bkg_MC"]:
        print(f"Loading background: {bkg_dir.relative_to(scored)} ({fold_label}) ...", flush=True)
        sc, wt = load_files(bkg_dir, test_fold_only=args.test_fold_only)
        print(f"  {len(sc)} events, sum(w) = {wt.sum():.4g}", flush=True)
        if len(sc):
            bkg_scores_list.append(sc)
            bkg_weights_list.append(wt)

    bkg_scores = np.concatenate(bkg_scores_list)
    bkg_weights = np.concatenate(bkg_weights_list)

    # keep only positive weights for significance
    sig_mask = sig_weights > 0
    bkg_mask = bkg_weights > 0

    thresholds = np.linspace(0.0, 1.0, 201)
    sig_yields, bkg_yields, significances = [], [], []

    for thr in thresholds:
        S = sig_weights[sig_mask & (sig_scores >= thr)].sum()
        B = bkg_weights[bkg_mask & (bkg_scores >= thr)].sum()
        sig_yields.append(S)
        bkg_yields.append(B)
        significances.append(asimov_significance(S, B))

    sig_yields = np.array(sig_yields)
    bkg_yields = np.array(bkg_yields)
    significances = np.array(significances)

    best_idx = np.argmax(significances)
    best_thr = thresholds[best_idx]
    best_z = significances[best_idx]
    best_s = sig_yields[best_idx]
    best_b = bkg_yields[best_idx]

    print(f"\nBest threshold: {best_thr:.3f}", flush=True)
    print(f"  S = {best_s:.4g}", flush=True)
    print(f"  B = {best_b:.4g}", flush=True)
    print(f"  S/sqrt(B) = {best_s/np.sqrt(best_b):.4f}" if best_b > 0 else "  S/sqrt(B) = N/A", flush=True)
    print(f"  Asimov Z  = {best_z:.4f}", flush=True)

    # --- inclusive (no GNN cut) ---
    S0 = sig_weights[sig_mask].sum()
    B0 = bkg_weights[bkg_mask].sum()
    Z0 = asimov_significance(S0, B0)
    print(f"\nInclusive (no GNN cut):", flush=True)
    print(f"  S = {S0:.4g}, B = {B0:.4g}, Z = {Z0:.4f}", flush=True)

    # --- 3-category scan: find optimal (t_low, t_high) ---
    print("\nScanning 3-category boundaries (t_low, t_high) ...", flush=True)
    cut_vals = np.linspace(0.05, 0.95, 91)  # 0.05, 0.06, ..., 0.95
    best3_z = 0.0
    best3_tlow = 0.6
    best3_thigh = 0.9
    for t_low in cut_vals:
        for t_high in cut_vals:
            if t_high <= t_low:
                continue
            z_comb, _, _ = three_cat_significance(
                sig_scores, sig_weights, bkg_scores, bkg_weights,
                sig_mask, bkg_mask, t_low, t_high)
            if z_comb > best3_z:
                best3_z = z_comb
                best3_tlow = t_low
                best3_thigh = t_high

    z_comb_best, zs_best, sb_best = three_cat_significance(
        sig_scores, sig_weights, bkg_scores, bkg_weights,
        sig_mask, bkg_mask, best3_tlow, best3_thigh)

    cat_labels = [f"score > {best3_thigh:.2f}",
                  f"{best3_tlow:.2f} < score <= {best3_thigh:.2f}",
                  f"score <= {best3_tlow:.2f}"]
    print(f"\nOptimal 3-category split: t_low={best3_tlow:.2f}, t_high={best3_thigh:.2f}", flush=True)
    for lbl, z, (S, B) in zip(cat_labels, zs_best, sb_best):
        print(f"  [{lbl}]  S={S:.4g}, B={B:.4g}, Z={z:.4f}", flush=True)
    print(f"  Combined Z = sqrt(sum Z_i^2) = {z_comb_best:.4f}", flush=True)

    # also report the fixed example split (0.6 / 0.9) for reference
    z_ref, zs_ref, sb_ref = three_cat_significance(
        sig_scores, sig_weights, bkg_scores, bkg_weights,
        sig_mask, bkg_mask, 0.6, 0.9)
    print(f"\nFixed split (0.6 / 0.9) for reference:", flush=True)
    for lbl, z, (S, B) in zip(["score > 0.9", "0.6 < score <= 0.9", "score <= 0.6"],
                               zs_ref, sb_ref):
        print(f"  [{lbl}]  S={S:.4g}, B={B:.4g}, Z={z:.4f}", flush=True)
    print(f"  Combined Z = {z_ref:.4f}", flush=True)

    # --- save metrics ---
    metrics = [
        f"data: Run 2 + Run 3 (2016preVFP/postVFP, 2017, 2018, 2022preEE/postEE, 2023preBPix/postBPix, 2024, 2025)",
        f"fold: {fold_label}",
        f"mass_window: [{MASS_LO}, {MASS_HI}] GeV",
        f"selection: is_SR == 1",
        f"",
        f"--- Single threshold (1-category) ---",
        f"inclusive_S: {S0:.8g}",
        f"inclusive_B: {B0:.8g}",
        f"inclusive_asimov_Z: {Z0:.6f}",
        f"best_threshold: {best_thr:.4f}",
        f"best_S: {best_s:.8g}",
        f"best_B: {best_b:.8g}",
        f"best_SoverSqrtB: {best_s/np.sqrt(best_b):.6f}" if best_b > 0 else "best_SoverSqrtB: N/A",
        f"best_asimov_Z: {best_z:.6f}",
        f"",
        f"--- 3-category (optimised) ---",
        f"3cat_t_low: {best3_tlow:.4f}",
        f"3cat_t_high: {best3_thigh:.4f}",
    ]
    for lbl, z, (S, B) in zip(cat_labels, zs_best, sb_best):
        metrics += [
            f"3cat_{lbl.replace(' ','_').replace('>','gt').replace('<','lt').replace('=','eq')}_S: {S:.8g}",
            f"3cat_{lbl.replace(' ','_').replace('>','gt').replace('<','lt').replace('=','eq')}_B: {B:.8g}",
            f"3cat_{lbl.replace(' ','_').replace('>','gt').replace('<','lt').replace('=','eq')}_Z: {z:.6f}",
        ]
    metrics += [
        f"3cat_combined_Z: {z_comb_best:.6f}",
        f"",
        f"--- 3-category (fixed 0.6/0.9) ---",
        f"3cat_fixed_combined_Z: {z_ref:.6f}",
    ]
    (output_dir / "significance.txt").write_text("\n".join(metrics) + "\n")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].plot(thresholds, significances, color="#2e4057", linewidth=2)
    axes[0].axvline(best_thr, color="crimson", linestyle="--", linewidth=1,
                    label=f"Best = {best_thr:.3f}\nZ = {best_z:.3f}")
    axes[0].axhline(z_comb_best, color="darkorange", linestyle=":", linewidth=1.5,
                    label=f"3-cat Z = {z_comb_best:.3f}\n(t_low={best3_tlow:.2f}, t_high={best3_thigh:.2f})")
    axes[0].set(xlabel="GNN score threshold", ylabel="Asimov significance Z")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(thresholds, sig_yields, color="#d1495b", linewidth=2, label="S (ttH)")
    axes[1].plot(thresholds, bkg_yields, color="#00798c", linewidth=2, label="B (all other)")
    axes[1].axvline(best_thr, color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(best3_tlow, color="darkorange", linestyle=":", linewidth=1.2, label=f"3-cat t_low={best3_tlow:.2f}")
    axes[1].axvline(best3_thigh, color="darkorange", linestyle="-.", linewidth=1.2, label=f"3-cat t_high={best3_thigh:.2f}")
    axes[1].set(xlabel="GNN score threshold", ylabel="Expected yield (weight_lumi)", yscale="log")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)

    # signal efficiency vs background rejection
    S_total = sig_yields[0] if sig_yields[0] > 0 else 1.0
    B_total = bkg_yields[0] if bkg_yields[0] > 0 else 1.0
    sig_eff = sig_yields / S_total
    bkg_rej = 1.0 - bkg_yields / B_total
    axes[2].plot(sig_eff, bkg_rej, color="#2e4057", linewidth=2)
    axes[2].scatter([sig_yields[best_idx] / S_total], [1.0 - bkg_yields[best_idx] / B_total],
                    color="crimson", zorder=5, label=f"Best thr = {best_thr:.3f}")
    axes[2].set(xlabel="Signal efficiency", ylabel="Background rejection", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"ttH vs other | Run 2+3 {fold_label} | m_llg ∈ [{MASS_LO}, {MASS_HI}] GeV | is_SR", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "significance.png", dpi=180)
    fig.savefig(output_dir / "significance.pdf")
    plt.close(fig)

    print(f"\nWrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
