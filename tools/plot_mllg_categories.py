#!/usr/bin/env python3
"""
plot_mllg_categories.py

Plot the m_llg distribution of the five analysis categories
(ttHhad, ttHlep, untag, VH3l, VHpTmiss) separately for the SR-tagged
and sideband-tagged events.

Category <-> source_tree_tag mapping (same as
significance_4class_4fold_cutbased.py):

    tag 0 -> ttHhad      tag 1 -> ttHhad sideband
    tag 2 -> ttHlep      tag 3 -> ttHlep sideband
    tag 4 -> untag       tag 5 -> untag sideband
    tag 6 -> VH3l        tag 7 -> VH3l sideband
    tag 8 -> VHpTmiss    tag 9 -> VHpTmiss sideband

Usage
-----
    python plot_mllg_categories.py
    python plot_mllg_categories.py --file <path/to/one/signal.root>
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
DEFAULT_FILE = (
    REPO
    / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold0"
    / "Signal"
    / "ttH"
    / "2018.root"
)
DEFAULT_OUTDIR = REPO / "tools" / "mllg_distributions"

TREE = "merged_SR_Sideband"
MASS = "m_llg_refit"
TAG = "source_tree_tag"
WEIGHT = "weight_lumi_avg"

CAT_TAGS = [
    ("ttHhad", 0),
    ("ttHlep", 2),
    ("untag", 4),
    ("VH3l", 6),
    ("VHpTmiss", 8),
]

# Analysis windows (same as the significance / GNN scripts).
SR_LO, SR_HI = 120.0, 130.0
SB_LO1, SB_HI1 = 110.0, 120.0
SB_LO2, SB_HI2 = 130.0, 180.0


def load(file_path, weight_branch, use_weight):
    with uproot.open(str(file_path)) as f:
        tree = f[TREE]
        branches = [MASS, TAG]
        if use_weight:
            branches.append(weight_branch)
        arrays = tree.arrays(branches, library="np")

    mass = np.asarray(arrays[MASS], dtype=np.float64)
    tag = np.asarray(arrays[TAG], dtype=np.int32)
    if use_weight:
        weight = np.asarray(arrays[weight_branch], dtype=np.float64)
    else:
        weight = np.ones_like(mass)

    good = np.isfinite(mass) & np.isfinite(weight)
    return mass[good], tag[good], weight[good]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--weight-branch", default=WEIGHT)
    parser.add_argument(
        "--no-weight", action="store_true",
        help="Ignore MC weights (unweighted event counts).",
    )
    parser.add_argument("--xmin", type=float, default=95.0)
    parser.add_argument("--xmax", type=float, default=180.0)
    parser.add_argument("--bin-width", type=float, default=2.0)
    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(args.file)

    args.outdir.mkdir(parents=True, exist_ok=True)
    use_weight = not args.no_weight

    mass, tag, weight = load(args.file, args.weight_branch, use_weight)

    bins = np.arange(args.xmin, args.xmax + 1e-6, args.bin_width)
    if bins[-1] < args.xmax:
        bins = np.append(bins, args.xmax)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    fig.suptitle(
        f"m_llg by category, SR (even tag) vs sideband (odd tag)\n"
        f"{args.file.name}  [{args.file.parent.parent.name}/{args.file.parent.name}]",
        fontsize=13,
    )

    hist_cache = {}
    rows = []
    for ax, (name, sr_tag) in zip(axes, CAT_TAGS):
        sb_tag = sr_tag + 1
        sr_sel = tag == sr_tag
        sb_sel = tag == sb_tag

        sr_m = mass[sr_sel]
        sb_m = mass[sb_sel]
        sr_w = weight[sr_sel]
        sb_w = weight[sb_sel]

        # Density (area-normalised) so the shapes can be compared.
        sr_h, _ = np.histogram(sr_m, bins=bins, weights=sr_w, density=True)
        sb_h, _ = np.histogram(sb_m, bins=bins, weights=sb_w, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        hist_cache[name] = (centers, sr_h, sb_h)

        ax.step(centers, sr_h, where="mid", color="tab:blue", lw=1.8,
                label=f"SR (tag {sr_tag}), N={sr_sel.sum()}")
        ax.step(centers, sb_h, where="mid", color="tab:orange", lw=1.8,
                ls="--", label=f"SB (tag {sb_tag}), N={sb_sel.sum()}")

        ax.axvspan(SR_LO, SR_HI, color="tab:blue", alpha=0.10, lw=0)
        ax.axvspan(SB_LO1, SB_HI1, color="tab:orange", alpha=0.08, lw=0)
        ax.axvspan(SB_LO2, SB_HI2, color="tab:orange", alpha=0.08, lw=0)

        ax.set_yscale("log")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

        rows.append({
            "category": name,
            "sr_tag": sr_tag,
            "sb_tag": sb_tag,
            "N_SR": int(sr_sel.sum()),
            "sumw_SR": float(np.sum(sr_w)),
            "N_SB": int(sb_sel.sum()),
            "sumw_SB": float(np.sum(sb_w)),
            "N_SR_in_mass_window": int(np.sum(
                (sr_m > SR_LO) & (sr_m < SR_HI)
            )),
        })

    for ax in axes[len(CAT_TAGS):]:
        ax.axis("off")
    for ax in axes[:len(CAT_TAGS)]:
        ax.set_xlabel("m_llg [GeV]")
    for ax in axes[0::3]:
        ax.set_ylabel("normalised events / bin")

    # Common y range so the five panels are directly comparable.
    all_vals = np.concatenate(
        [h[1] for h in hist_cache.values()] + [h[2] for h in hist_cache.values()]
    )
    positive = all_vals[all_vals > 0]
    y_lo = 10 ** np.floor(np.log10(positive.min())) * 0.7
    y_hi = 10 ** np.ceil(np.log10(positive.max())) * 1.6
    for ax in axes[:len(CAT_TAGS)]:
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = args.outdir / "mllg_sr_vs_sb_5cat"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    fig.savefig(stem.with_suffix(".pdf"))
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")

    # Second figure: all five categories overlaid, SR and SB side by side.
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    colors = plt.cm.tab10(np.arange(len(CAT_TAGS)))
    for ax, idx, label in ((axes2[0], 1, "SR (even tag)"),
                           (axes2[1], 2, "sideband (odd tag)")):
        for color, (name, _) in zip(colors, CAT_TAGS):
            centers, sr_h, sb_h = hist_cache[name]
            ax.step(centers, sr_h if idx == 1 else sb_h, where="mid",
                    color=color, lw=1.6, label=name)
        ax.axvspan(SR_LO, SR_HI, color="tab:blue", alpha=0.10, lw=0)
        ax.axvspan(SB_LO1, SB_HI1, color="tab:orange", alpha=0.08, lw=0)
        ax.axvspan(SB_LO2, SB_HI2, color="tab:orange", alpha=0.08, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("m_llg [GeV]")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes2[0].set_ylabel("normalised events / bin")
    axes2[0].legend(fontsize=8)
    fig2.suptitle(f"{args.file.name}: all categories overlaid", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    stem2 = args.outdir / "mllg_all_categories_overlay"
    fig2.savefig(stem2.with_suffix(".png"), dpi=150)
    fig2.savefig(stem2.with_suffix(".pdf"))
    print(f"Saved {stem2.with_suffix('.png')}")
    print(f"Saved {stem2.with_suffix('.pdf')}")

    # Also write the histogram contents for inspection.
    csv_path = args.outdir / "mllg_sr_vs_sb_5cat.csv"
    header = ["bin_lo", "bin_hi"]
    for name, _ in CAT_TAGS:
        header += [f"{name}_SR_density", f"{name}_SB_density"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(len(bins) - 1):
            row = [bins[i], bins[i + 1]]
            for name, sr_tag in CAT_TAGS:
                for sel, wgt in ((tag == sr_tag, weight),
                                 (tag == sr_tag + 1, weight)):
                    h, _ = np.histogram(mass[sel], bins=bins,
                                        weights=wgt[sel], density=True)
                    row.append(h[i])
            writer.writerow(row)
    print(f"Saved {csv_path}")

    print("\nCategory summary")
    print(f"{'category':<10s}{'tag':>4s}{'N_SR':>9s}{'sumw_SR':>12s}"
          f"{'N_SB':>8s}{'sumw_SB':>12s}{'SR_in_window':>14s}")
    for r in rows:
        print(f"{r['category']:<10s}{r['sr_tag']:>4d}{r['N_SR']:>9d}"
              f"{r['sumw_SR']:>12.4f}{r['N_SB']:>8d}{r['sumw_SB']:>12.4f}"
              f"{r['N_SR_in_mass_window']:>14d}")


if __name__ == "__main__":
    main()
