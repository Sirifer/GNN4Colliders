#!/usr/bin/env python3
"""Compare the cut-based and 4-class GNN H->Zgamma significance results.

Reads the CSV outputs of

    tools/significance_4fold_cutbase.py       (5 cut categories, common alpha)
    tools/significance_4class_v3_bkg_4cat.py  (4 GNN categories, common alpha)

Both scripts use the same events, the same MC weight branch and the same
sideband normalisation, so the total S and B per fold must agree.  That is
asserted here; the Z difference is therefore caused by the binning only.

Outputs (tools/results_compare_cutbase_vs_gnn/):
    report.md
    per_fold_comparison.csv
    per_category_comparison.csv
    comparison.png

Usage:
    python tools/compare_cutbase_vs_gnn.py
"""

import csv
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
TOOLS = REPO / "tools"

CUT_CSV = TOOLS / "results_4fold_cutbase" / "category_yields.csv"
GNN_CSV = TOOLS / "results_4class_v3_bkg_4cat" / "category_yields.csv"
PER_CAT_ALPHA_TXT = TOOLS / "results_4class_4fold_cutbased" / "final_z.txt"

OUT_DIR = TOOLS / "results_compare_cutbase_vs_gnn"

CUT_ORDER = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]
GNN_ORDER = ["Other", "ttH", "VH", "ggFVBF"]

# Validated light-mode categorical pair (dataviz palette slots 1, 2).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
CUT_COLOR = "#2a78d6"
GNN_COLOR = "#eb6834"
REF_COLOR = "#b8b7b2"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def asimov_z(s, b):
    if not np.isfinite(s) or not np.isfinite(b) or s <= 0.0 or b <= 0.0:
        return 0.0
    return float(np.sqrt(max(2.0 * ((s + b) * np.log1p(s / b) - s), 0.0)))


def combine_cut(rows):
    """Reproduce significance_4fold_cutbase.combined_scaled()."""
    z = float(np.sqrt(sum(r["Z"] ** 2 for r in rows)))
    if z <= 0.0:
        return 0.0, 0.0
    sigma = float(np.sqrt(sum((r["Z"] / z * r["sigma_Z"]) ** 2 for r in rows)))
    return z, sigma


def combine_gnn(rows, alpha, sigma_alpha):
    """Reproduce significance_4class_v3_bkg_4cat.combined_z()."""
    z = float(np.sqrt(sum(r["Z"] ** 2 for r in rows)))
    if z <= 0.0:
        return 0.0, 0.0

    var = 0.0
    alpha_derivative_sum = 0.0
    for r in rows:
        s, b = r["S"], r["B"]
        if s <= 0.0 or b <= 0.0:
            continue
        log_term = np.log1p(s / b)
        dzt_ds = log_term / z
        dzt_db = (log_term - s / b) / z
        var += (dzt_ds * r["sigma_S"]) ** 2
        var += (dzt_db * alpha * r["sigma_Y"]) ** 2
        alpha_derivative_sum += dzt_db * r["Y"]
    var += (alpha_derivative_sum * sigma_alpha) ** 2
    return z, float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _f(row, key):
    return float(row[key])


def load_cut():
    rows = []
    with CUT_CSV.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows.append({
                "fold": int(r["fold"]),
                "category": r["category"],
                "tag": int(r["source_tree_tag"]),
                "alpha": _f(r, "alpha"),
                "sigma_alpha": _f(r, "sigma_alpha"),
                "S": _f(r, "S"),
                "sigma_S": _f(r, "sigma_S"),
                "Y": _f(r, "B_mc_sr"),
                "sigma_Y": None,
                "B": _f(r, "B"),
                "sigma_B": _f(r, "sigma_B"),
                "Z": _f(r, "Z"),
                "sigma_Z": _f(r, "sigma_Z"),
            })
    return rows


def load_gnn(split="test"):
    rows = []
    with GNN_CSV.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["split"] != split:
                continue
            rows.append({
                "fold": int(r["fold"]),
                "category": r["category"],
                "tag": None,
                "alpha": _f(r, "alpha"),
                "sigma_alpha": _f(r, "sigma_alpha"),
                "S": _f(r, "S"),
                "sigma_S": _f(r, "sigma_S"),
                "Y": _f(r, "Y_bkg_sr"),
                "sigma_Y": _f(r, "sigma_Y_bkg_sr"),
                "B": _f(r, "B"),
                "sigma_B": _f(r, "sigma_B"),
                "Z": _f(r, "Z"),
                "sigma_Z": _f(r, "sigma_Z"),
            })
    return rows


def by_fold(rows):
    return {f: [r for r in rows if r["fold"] == f] for f in range(4)}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def main():
    cut_rows = load_cut()
    gnn_rows = load_gnn("test")
    gnn_val_rows = load_gnn("validation")

    cut_folds = by_fold(cut_rows)
    gnn_folds = by_fold(gnn_rows)
    gnn_val_folds = by_fold(gnn_val_rows)

    folds = []
    for k in range(4):
        c = cut_folds[k]
        g = gnn_folds[k]
        s_cut = sum(r["S"] for r in c)
        b_cut = sum(r["B"] for r in c)
        s_gnn = sum(r["S"] for r in g)
        b_gnn = sum(r["B"] for r in g)

        if not np.isclose(s_cut, s_gnn, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"fold {k}: signal totals differ ({s_cut} vs {s_gnn})")
        if not np.isclose(b_cut, b_gnn, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"fold {k}: background totals differ ({b_cut} vs {b_gnn})")

        alpha_cut = c[0]["alpha"]
        alpha_gnn = g[0]["alpha"]
        if not np.isclose(alpha_cut, alpha_gnn, rtol=1e-9, atol=0.0):
            raise RuntimeError(f"fold {k}: alpha differs ({alpha_cut} vs {alpha_gnn})")

        z_cut, sig_cut = combine_cut(c)
        z_gnn, sig_gnn = combine_gnn(g, alpha_gnn, g[0]["sigma_alpha"])
        z_ref = asimov_z(s_cut, b_cut)

        # Fraction of the total signal that lands in the most bkg-dominated bin.
        dominant_cut = max(c, key=lambda r: r["B"])
        dominant_gnn = max(g, key=lambda r: r["B"])

        folds.append({
            "fold": k,
            "alpha": alpha_cut,
            "sigma_alpha": c[0]["sigma_alpha"],
            "S": s_cut,
            "B": b_cut,
            "SB": s_cut / b_cut if b_cut > 0 else 0.0,
            "Z_cut": z_cut,
            "sigmaZ_cut": sig_cut,
            "Z_gnn": z_gnn,
            "sigmaZ_gnn": sig_gnn,
            "Z_ref": z_ref,
            "ratio_gnn_cut": z_gnn / z_cut if z_cut > 0 else 0.0,
            "gain_cut_ref": z_cut / z_ref - 1.0 if z_ref > 0 else 0.0,
            "gain_gnn_ref": z_gnn / z_ref - 1.0 if z_ref > 0 else 0.0,
            "dom_cut_cat": dominant_cut["category"],
            "dom_cut_Sfrac": dominant_cut["S"] / s_cut if s_cut > 0 else 0.0,
            "dom_cut_Bfrac": dominant_cut["B"] / b_cut if b_cut > 0 else 0.0,
            "dom_gnn_cat": dominant_gnn["category"],
            "dom_gnn_Sfrac": dominant_gnn["S"] / s_gnn if s_gnn > 0 else 0.0,
            "dom_gnn_Bfrac": dominant_gnn["B"] / b_gnn if b_gnn > 0 else 0.0,
        })

    z_cut_total = float(np.sqrt(sum(f["Z_cut"] ** 2 for f in folds)))
    z_gnn_total = float(np.sqrt(sum(f["Z_gnn"] ** 2 for f in folds)))
    z_ref_total = float(np.sqrt(sum(f["Z_ref"] ** 2 for f in folds)))
    sig_cut_total = float(np.sqrt(sum(
        (f["Z_cut"] / z_cut_total * f["sigmaZ_cut"]) ** 2 for f in folds)))
    sig_gnn_total = float(np.sqrt(sum(
        (f["Z_gnn"] / z_gnn_total * f["sigmaZ_gnn"]) ** 2 for f in folds)))

    # Validation Z (GNN only; the cut-based script has no validation split).
    z_gnn_val_total = float(np.sqrt(sum(
        combine_gnn(gnn_val_folds[k], gnn_val_folds[k][0]["alpha"],
                    gnn_val_folds[k][0]["sigma_alpha"])[0] ** 2
        for k in range(4)
    )))

    # Category aggregates over the four disjoint test parities.
    def aggregate(rows, order):
        out = []
        z_total = float(np.sqrt(sum(r["Z"] ** 2 for r in rows)))
        for name in order:
            sub = [r for r in rows if r["category"] == name]
            z_cat = float(np.sqrt(sum(r["Z"] ** 2 for r in sub)))
            out.append({
                "category": name,
                "S": sum(r["S"] for r in sub),
                "sigma_S": float(np.sqrt(sum(r["sigma_S"] ** 2 for r in sub))),
                "B": sum(r["B"] for r in sub),
                "sigma_B": float(np.sqrt(sum(r["sigma_B"] ** 2 for r in sub))),
                "Z": z_cat,
                "sigma_Z": float(np.sqrt(sum(r["sigma_Z"] ** 2 for r in sub))),
                "z2_share": z_cat ** 2 / z_total ** 2 if z_total > 0 else 0.0,
                "Sfrac": (sum(r["S"] for r in sub)
                          / sum(r["S"] for r in rows)),
                "Bfrac": (sum(r["B"] for r in sub)
                          / sum(r["B"] for r in rows)),
            })
        return out

    cut_cats = aggregate(cut_rows, CUT_ORDER)
    gnn_cats = aggregate(gnn_rows, GNN_ORDER)

    # Optional appendix number: per-category-alpha cut-based variant.
    per_cat_alpha_z = None
    if PER_CAT_ALPHA_TXT.exists():
        text = PER_CAT_ALPHA_TXT.read_text(encoding="utf-8")
        match = re.search(
            r"4-fold combined Test Z\s*=\s*([0-9.]+)", text)
        if match:
            per_cat_alpha_z = float(match.group(1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csvs(folds, cut_cats, gnn_cats)
    make_figure(folds, cut_cats, gnn_cats)
    write_report(
        folds, cut_cats, gnn_cats,
        z_cut_total, sig_cut_total, z_gnn_total, sig_gnn_total,
        z_ref_total, z_gnn_val_total, per_cat_alpha_z,
    )

    print(f"Report: {OUT_DIR / 'report.md'}")
    print(f"Figure: {OUT_DIR / 'comparison.png'}")
    print(f"4-fold Z: cut = {z_cut_total:.6f}, GNN = {z_gnn_total:.6f}, "
          f"no-binning = {z_ref_total:.6f}")
    print(f"GNN / cut = {z_gnn_total / z_cut_total:.4f}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_csvs(folds, cut_cats, gnn_cats):
    with (OUT_DIR / "per_fold_comparison.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(folds[0]))
        writer.writeheader()
        writer.writerows(folds)

    fields = ["scheme", "category", "S", "sigma_S", "B", "sigma_B",
              "S_over_B", "Z", "sigma_Z", "z2_share", "Sfrac", "Bfrac"]
    with (OUT_DIR / "per_category_comparison.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scheme, cats in (("cut-based", cut_cats), ("gnn-4class", gnn_cats)):
            for r in cats:
                writer.writerow({
                    "scheme": scheme,
                    "category": r["category"],
                    "S": r["S"],
                    "sigma_S": r["sigma_S"],
                    "B": r["B"],
                    "sigma_B": r["sigma_B"],
                    "S_over_B": r["S"] / r["B"] if r["B"] > 0 else 0.0,
                    "Z": r["Z"],
                    "sigma_Z": r["sigma_Z"],
                    "z2_share": r["z2_share"],
                    "Sfrac": r["Sfrac"],
                    "Bfrac": r["Bfrac"],
                })


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(folds, cut_cats, gnn_cats,
                 z_cut, sig_cut, z_gnn, sig_gnn, z_ref,
                 z_gnn_val, per_cat_alpha_z):
    ratio = z_gnn / z_cut if z_cut > 0 else 0.0

    summary = md_table(
        ["Quantity", "Cut-based (5 cat)", "GNN 4-class", "GNN / cut"],
        [
            ["4-fold combined test Z", f"{z_cut:.6f}", f"{z_gnn:.6f}",
             f"{ratio:.4f}"],
            ["reported error on Z", f"± {sig_cut:.6f}", f"± {sig_gnn:.6f}",
             "not comparable*"],
            ["no-binning reference Z", f"{z_ref:.6f}", f"{z_ref:.6f}",
             "identical"],
            ["binning gain vs no-binning",
             f"{z_cut / z_ref - 1:+.2%}" if z_ref > 0 else "n/a",
             f"{z_gnn / z_ref - 1:+.2%}" if z_ref > 0 else "n/a", "—"],
        ],
    )

    fold_rows = []
    for f in folds:
        fold_rows.append([
            f["fold"],
            f"{f['alpha']:.6f}",
            f"{f['S']:.6f}",
            f"{f['B']:.4f}",
            f"{f['Z_cut']:.6f}",
            f"{f['Z_gnn']:.6f}",
            f"{f['Z_gnn'] - f['Z_cut']:+.6f}",
            f"{f['Z_ref']:.6f}",
        ])
    fold_table = md_table(
        ["Fold", "alpha", "S", "B", "Z cut", "Z GNN", "ΔZ (GNN−cut)",
         "Z no-binning"],
        fold_rows,
    )

    def cat_rows(cats):
        return [[
            r["category"],
            f"{r['S']:.4f}",
            f"{r['B']:.2f}",
            f"{(r['S'] / r['B'] if r['B'] > 0 else 0):.4f}",
            f"{r['Z']:.5f}",
            f"{r['z2_share']:.1%}",
            f"{r['Sfrac']:.1%}",
            f"{r['Bfrac']:.1%}",
        ] for r in cats]

    cat_header = ["Category", "S", "B", "S/B", "Z (4 folds)",
                  "Z² share", "signal fraction", "bkg fraction"]
    cut_table = md_table(cat_header, cat_rows(cut_cats))
    gnn_table = md_table(cat_header, cat_rows(gnn_cats))

    dom_cut_S = np.mean([f["dom_cut_Sfrac"] for f in folds])
    dom_cut_B = np.mean([f["dom_cut_Bfrac"] for f in folds])
    dom_gnn_S = np.mean([f["dom_gnn_Sfrac"] for f in folds])
    dom_gnn_B = np.mean([f["dom_gnn_Bfrac"] for f in folds])
    dom_cut_name = folds[0]["dom_cut_cat"]
    dom_gnn_name = folds[0]["dom_gnn_cat"]

    appendix = ""
    if per_cat_alpha_z is not None:
        appendix = (
            "\n## Appendix: per-category-alpha cut-based variant\n\n"
            "`tools/significance_4class_4fold_cutbased.py` uses "
            f"`alpha_i = N(Data SB_i) / Y(Bkg MC SB_i)` per category and "
            f"reports combined test Z = {per_cat_alpha_z:.6f}. "
            "That is a **different normalisation** from the common-alpha "
            "results above and is shown only for bookkeeping; do not place "
            "it in the same column as the numbers in this report.\n"
        )

    report = f"""# Cut-based vs 4-class GNN significance comparison

Generated by `tools/compare_cutbase_vs_gnn.py` from the committed CSV
outputs; no ROOT file is re-read. Both inputs use the same events, the same
`weight_lumi_avg` MC weights and one common sideband scale factor per fold.

- Cut-based: `tools/significance_4fold_cutbase.py`
  (5 `source_tree_tag` categories, common alpha, test parity only)
- GNN 4-class: `tools/significance_4class_v3_bkg_4cat.py`
  (4 argmax categories, common alpha per fold/split, cuts tuned on validation)

## Summary

{summary}

\\*The two scripts propagate the shared-alpha correlation differently
(coherent sum in the GNN script, per-category quadrature in the cut-based
script), so the quoted errors are not on the same footing. The central Z
values are directly comparable.

## Per-fold test results

{fold_table}

The four test parities are disjoint, so the 4-fold number is
`sqrt(sum_k Z_k^2)`. `alpha`, total `S` and total `B` agree between the two
schemes to the precision printed — the difference in Z is caused by the
binning alone.

## Cut-based categories

{cut_table}

## GNN categories

{gnn_table}

## Where the gain comes from

Both schemes see the same total signal and background, and the binned Asimov
significance

    Z = sqrt( sum_i 2 [ (s_i + b_i) ln(1 + s_i/b_i) - s_i ] )

is largest when signal sits in bins with small `b_i`. The difference is
therefore how much signal is left in the background-dominated bin:

| Scheme | background-dominated bin | mean signal fraction there | mean bkg fraction there |
|---|---|---|---|
| cut-based | {dom_cut_name} | {dom_cut_S:.1%} | {dom_cut_B:.1%} |
| GNN 4-class | {dom_gnn_name} | {dom_gnn_S:.1%} | {dom_gnn_B:.1%} |

The GNN cuts the fraction of signal left in the background-dominated bin
from {dom_cut_S:.1%} to {dom_gnn_S:.1%} and redistributes it into the
`ttH`, `VH` and `ggFVBF` bins, which raises the combined Z from
{z_cut:.4f} to {z_gnn:.4f} (+{100 * ratio - 100:.2f}%). The fraction of
background in that dominant bin is essentially unchanged
({dom_cut_B:.1%} → {dom_gnn_B:.1%}), so the gain is a signal-purity effect,
not a background-rejection effect.

For reference, the GNN validation-fold combination (cuts tuned on that
parity) gives Z = {z_gnn_val:.4f}; only the test Z is used in the headline
number.

## Caveats

1. The cut-based script has no validation/test split — it evaluates every
   test parity with the same fixed tag selection. The GNN cuts are tuned on
   the validation parity, which for fold `k` is part of the model's training
   set; the test parity is still held out.
2. The two Z values are not statistically independent and must not be added
   or combined.
3. `source_tree_tag` already encodes `m_llg_refit` in (120, 130) for even
   tags, so neither script applies another mass window to the SR. Odd tags
   also contain a few percent to ~15% of events outside the nominal
   110–120 / 130–180 GeV SB range (sample-dependent); both scripts use the
   same odd-tag definition, so alpha matches.
4. The background model is MC-scaled by one data-driven alpha per fold; no
   signal contamination in the sideband and no profile-likelihood fit is
   included.
{appendix}
## Files

- `per_fold_comparison.csv` — per-fold Z, yields, alpha, gains
- `per_category_comparison.csv` — aggregated category yields and Z
- `comparison.png` — figure below

![cut-based vs GNN comparison](comparison.png)
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(False)


def make_figure(folds, cut_cats, gnn_cats):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.25),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )
    fig.patch.set_facecolor(SURFACE)

    # (a) Per-fold test Z ---------------------------------------------------
    ax = axes[0]
    _style_axis(ax)
    x = np.arange(len(folds))
    width = 0.26
    cut_z = [f["Z_cut"] for f in folds]
    gnn_z = [f["Z_gnn"] for f in folds]
    ref_z = [f["Z_ref"] for f in folds]

    ax.bar(x - width, cut_z, width, color=CUT_COLOR,
           edgecolor=SURFACE, linewidth=1.0, label="Cut-based (5 cat)")
    ax.bar(x, gnn_z, width, color=GNN_COLOR,
           edgecolor=SURFACE, linewidth=1.0, label="GNN 4-class")
    ax.bar(x + width, ref_z, width, color=REF_COLOR,
           edgecolor=SURFACE, linewidth=1.0,
           label="No binning (total S, B)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f['fold']}" for f in folds])
    ax.set_ylabel("Asimov Z (test parity)")
    ax.set_title("(a) Significance per fold", loc="left",
                 color=INK, fontsize=10.5, pad=8)
    ax.set_ylim(0.0, max(max(cut_z), max(gnn_z), max(ref_z)) * 1.22)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handlelength=1.0, borderaxespad=0.4)

    # (b), (c) Category contributions --------------------------------------
    panels = [
        (axes[1], cut_cats, CUT_COLOR, "(b) Cut-based categories"),
        (axes[2], gnn_cats, GNN_COLOR, "(c) GNN categories"),
    ]
    xmax = max(max(r["Z"] for r in cut_cats), max(r["Z"] for r in gnn_cats))
    pad = xmax * 0.035

    for ax, cats, color, title in panels:
        _style_axis(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=1.0)

        y = np.arange(len(cats))
        zvals = [r["Z"] for r in cats]
        ax.barh(y, zvals, height=0.55, color=color,
                edgecolor=SURFACE, linewidth=1.0)
        for yi, r in zip(y, cats):
            ax.text(r["Z"] + pad, yi, f"{r['Z']:.3f}",
                    va="center", ha="left", fontsize=8, color=INK2)

        ax.set_yticks(y)
        ax.set_yticklabels([r["category"] for r in cats])
        ax.invert_yaxis()
        ax.set_xlim(0.0, xmax * 1.20)
        ax.set_xlabel("Z (four test parities combined)")
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=8)

    fig.suptitle(
        "Cut-based vs GNN 4-class: identical events and normalisation, "
        "different binning",
        x=0.012, ha="left", fontsize=12, color=INK, y=1.0,
    )
    fig.text(
        0.012, 0.925,
        "H→Zγ v3, four-fold test parities. Both schemes share total S, "
        "total B and the common sideband alpha.",
        ha="left", fontsize=8.5, color=INK2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "comparison.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
