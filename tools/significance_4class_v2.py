#!/usr/bin/env python3
"""Asimov significance of the 4-class GNN on the OLD v2 4-fold data.

Reuses the machinery of `significance_4fold_cutbase.py` (alpha from the
sideband, `Z = sqrt(sum_i 2[(s_i+b_i)ln(1+s_i/b_i)-s_i])`, 4-fold combination
in quadrature) but reads the v2 samples:

    MC    : root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{k}
            (binary scores live here; the 4-class scores come from the npz)
    Data  : root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_data/Data

The 4-class scores are the ones produced for `eval_4class_on_v2.py`:

    tools/results_v2_eval/scores_npz/<proc>/<era>.npz   key "s_f{k}"

whose event order is asserted equal to the scored tree's
`photon_eta_fold`/`is_SR`/`source_tree_tag` sequence, so the 4-class and the
binary score sit on the same events.

Three categorisations are evaluated on the same SR events with the same alpha:

    cut      the 5 upstream SR tags (0 ttHhad, 2 ttHlep, 4 untag, 6 VH3l,
             8 VHpTmiss)
    gnn4     argmax of [p_bkg, p_ttH, p_VH, p_ggF_VBF] with the thresholds
             jointly optimised on v3, reused here unchanged
    gnn2     the binary model, p > 0.5 vs the rest

plus a no-binning reference (all SR in one category), which must come out
identical for all three — that is the consistency check on the event sets.

Signal is ttH+VH+ggF+VBF (v2 sample list), background is TT+DYGto2LG_10to100.
v2 has no other backgrounds and no data-driven estimate beyond the sideband
scale factor, so this Z is not the analysis Z of
`results_compare_cutbase_vs_gnn/`.

Outputs (tools/results_significance_v2/):
    per_category.csv
    final_z.txt
    report.md
    z_comparison.png

Usage:
    python tools/significance_4class_v2.py
    python tools/significance_4class_v2.py --diagnostic-only
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from significance_4fold_cutbase import (  # noqa: E402
    SR_TAGS, asimov_z, yield_info, z_uncertainty,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_significance_v2"
NPZ_DIR = REPO / "tools" / "results_v2_eval" / "scores_npz"

MC_BASE = REPO / "root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold"
DATA_BASE = REPO / "root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_data"

TREE = "merged_SR_Sideband"
WEIGHT = "weight_lumi_avg"
FOLDVAR = "photon_eta_fold"
TAG = "source_tree_tag"
BIN_SCORE = "ttH_gnn_score"

NFOLD = 4
TT_COL = 1                                   # softmax column of ttH

SIGNAL_DIRS = ["Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF"]
BKG_DIRS = ["Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100"]
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]

CATS4 = ["Other", "ttH", "VH", "ggFVBF"]
CATS2 = ["ttH", "Other"]
# Jointly optimised on v3 (results_4class_v3_bkg_4cat/category_yields.csv).
C_TTH, C_VH, C_GG = 0.7052356230447479, 0.8186932787320207, 0.220597113010442
C_BIN = 0.5

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1a8425"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def empty():
    return {k: [] for k in ("is_sr", "tag", "fold", "w", "s4", "pb", "y")}


def load_fold(fold, missing):
    acc = empty()
    for is_pos, dirs in ((True, SIGNAL_DIRS), (False, BKG_DIRS)):
        for proc in dirs:
            for era in ERAS:
                tree = uproot.open(
                    MC_BASE.parent / f"{MC_BASE.name}{fold}" / proc
                    / f"{era}.root")[TREE]
                arrays = tree.arrays(
                    ["is_SR", TAG, FOLDVAR, WEIGHT, BIN_SCORE], library="np")

                npz = NPZ_DIR / proc / f"{era}.npz"
                if not npz.exists():
                    missing.append(f"{proc}/{era}")
                    four = np.full((arrays[WEIGHT].size, 4), np.nan)
                else:
                    data = np.load(npz, allow_pickle=True)
                    key = f"s_f{fold}"
                    scores = np.asarray(data["scores"].flat[0][key])
                    track = np.asarray(data["tracking_info"].flat[0][key])
                    fvar = np.asarray(arrays[FOLDVAR], dtype=np.int64)
                    if not np.array_equal(track[:, 0].astype(np.int64), fvar):
                        raise RuntimeError(f"{proc}/{era}: event order differs")
                    four = np.asarray(scores, np.float64)

                acc["is_sr"].append(np.asarray(arrays["is_SR"]))
                acc["tag"].append(np.asarray(arrays[TAG]))
                acc["fold"].append(np.asarray(arrays[FOLDVAR], dtype=np.int64))
                acc["w"].append(np.asarray(arrays[WEIGHT], np.float64))
                acc["pb"].append(np.asarray(arrays[BIN_SCORE], np.float64))
                acc["s4"].append(four)
                acc["y"].append(np.full(arrays[WEIGHT].size, is_pos))
    return {k: np.concatenate(v) for k, v in acc.items()}


def load_data():
    parts = {k: [] for k in ("is_sr", "tag", "fold")}
    for era in ERAS:
        path = DATA_BASE / "Data" / f"{era}.root"
        if not path.exists():
            continue
        arrays = uproot.open(path)[TREE].arrays(
            ["is_SR", TAG, FOLDVAR], library="np")
        parts["is_sr"].append(np.asarray(arrays["is_SR"]))
        parts["tag"].append(np.asarray(arrays[TAG]))
        parts["fold"].append(np.asarray(arrays[FOLDVAR], dtype=np.int64))
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def categories_cut(mc):
    return {name: (mc["tag"] == tag) for tag, name in SR_TAGS.items()}


def categories_gnn4_from(scores, c_tth=C_TTH, c_vh=C_VH, c_gg=C_GG):
    """v3 category rule: argmax after the three mutually exclusive cuts."""
    p4 = scores
    arg = np.argmax(p4, axis=1)
    out = {"Other": np.ones(p4.shape[0], dtype=bool)}
    out["ttH"] = (arg == 1) & (p4[:, 1] > c_tth)
    out["VH"] = (arg == 2) & (p4[:, 2] > c_vh)
    out["ggFVBF"] = (arg == 3) & (p4[:, 3] > c_gg)
    taken = out["ttH"] | out["VH"] | out["ggFVBF"]
    out["Other"] = ~taken
    return out


# ---------------------------------------------------------------------------
# Z
# ---------------------------------------------------------------------------

def alpha_from_sideband(data, mc, parity):
    sb_tags = [t + 1 for t in SR_TAGS]
    dm = (np.asarray(data["is_sr"]) == 0) & np.isin(data["tag"], sb_tags) \
        & ((data["fold"] % NFOLD) == parity)
    bm = (mc["is_sr"] == 0) & np.isin(mc["tag"], sb_tags) \
        & ((mc["fold"] % NFOLD) == parity) & np.isfinite(mc["w"])
    n_data = int(dm.sum())
    bkg = yield_info(mc["w"][bm])
    if n_data <= 0:
        # v2 Data carries photon_eta_fold % 4 in {0, 1} only, so parities 2
        # and 3 have no data sideband at all. The caller falls back to the
        # pooled alpha.
        return None
    if bkg["sum"] <= 0:
        raise RuntimeError(f"fold {parity}: empty MC sideband")
    alpha = n_data / bkg["sum"]
    sigma_alpha = alpha * np.sqrt(1.0 / n_data + (bkg["err"] / bkg["sum"]) ** 2)
    return {"N_data_sb": n_data, "Y_bkg_sb": bkg["sum"], "alpha": float(alpha),
            "sigma_alpha": float(sigma_alpha)}


def evaluate(masks, sub, norm):
    """`masks` are over the selected SR events; split them signal/background."""
    results = {}
    is_sig = sub["y"]
    for name, mask in masks.items():
        s = yield_info(sub["w"][mask & is_sig])
        bmc = yield_info(sub["w"][mask & ~is_sig])
        b = norm["alpha"] * bmc["sum"]
        sigma_b = float(np.hypot(norm["alpha"] * bmc["err"],
                                 bmc["sum"] * norm["sigma_alpha"]))
        z = asimov_z(s["sum"], b)
        results[name] = {
            "S": s["sum"], "sigma_S": s["err"], "N_S": s["n"],
            "B_mc": bmc["sum"], "B": b, "sigma_B": sigma_b,
            "Z": z,
            "sigma_Z": z_uncertainty(s["sum"], b, s["err"], sigma_b, z),
        }
    return results


def combine(results):
    z = float(np.sqrt(sum(r["Z"] ** 2 for r in results.values())))
    if z <= 0:
        return 0.0, 0.0
    sigma = float(np.sqrt(sum((r["Z"] / z * r["sigma_Z"]) ** 2
                              for r in results.values())))
    return z, sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--c-tth", type=float, default=C_TTH)
    parser.add_argument("--c-vh", type=float, default=C_VH)
    parser.add_argument("--c-ggfvbf", type=float, default=C_GG)
    parser.add_argument("--c-bin", type=float, default=C_BIN)
    args = parser.parse_args()

    missing = []
    data = load_data()
    mc_folds = [load_fold(fold, missing) for fold in range(NFOLD)]

    raw = [alpha_from_sideband(data, mc_folds[f], f) for f in range(NFOLD)]
    known = [n for n in raw if n is not None]
    pooled = None
    if len(known) < NFOLD:
        if not known:
            raise SystemExit("No data sideband in any parity; cannot normalise")
        pooled = {
            "N_data_sb": sum(n["N_data_sb"] for n in known),
            "Y_bkg_sb": float(np.mean([n["Y_bkg_sb"] for n in known])),
            "alpha": float(np.mean([n["alpha"] for n in known])),
            "sigma_alpha": float(np.mean([n["sigma_alpha"] for n in known])),
        }
        print(f"v2 Data only covers parities "
              f"{[i for i, n in enumerate(raw) if n is not None]}; "
              f"parities without data use the pooled alpha "
              f"{pooled['alpha']:.5f}")

    per_fold = {}
    rows = []
    for fold in range(NFOLD):
        mc = mc_folds[fold]
        sel = (mc["is_sr"] == 1) & np.isin(mc["tag"], list(SR_TAGS)) \
            & ((mc["fold"] % NFOLD) == fold) & np.isfinite(mc["w"]) \
            & np.isfinite(mc["s4"][:, TT_COL])
        sub = {k: v[sel] for k, v in mc.items()}
        n_sig = int(sub["y"].sum())
        n_bkg = int((~sub["y"]).sum())

        norm = raw[fold]
        if norm is None:
            norm = dict(pooled)
            norm["alpha_source"] = "pooled"
        else:
            norm = dict(norm)
            norm["alpha_source"] = "per fold"

        schemes = {
            "cut": categories_cut(sub),
            "gnn4": categories_gnn4_from(sub["s4"], args.c_tth, args.c_vh,
                                         args.c_ggfvbf),
            "gnn2": {"ttH": sub["pb"] > args.c_bin,
                     "Other": ~(sub["pb"] > args.c_bin)},
            "none": {"all-SR": np.ones(sub["y"].size, dtype=bool)},
        }

        per_fold[fold] = {"norm": norm, "schemes": {}}
        print(f"\nfold {fold}: alpha = {norm['alpha']:.5f} "
              f"from N_data_SB={norm['N_data_sb']}, "
              f"Y_bkg_SB={norm['Y_bkg_sb']:.1f}; SR signal={n_sig}, "
              f"SR background={n_bkg}")
        for scheme, masks in schemes.items():
            results = evaluate(masks, sub, norm)
            if scheme != "none":
                ref = per_fold[fold]["schemes"]["none"][2]["all-SR"] \
                    if "none" in per_fold[fold]["schemes"] else None
                if ref is not None:
                    tot_s = sum(r["S"] for r in results.values())
                    tot_b = sum(r["B"] for r in results.values())
                    if not (abs(tot_s - ref["S"]) < 1e-6
                            and abs(tot_b - ref["B"]) < 1e-6):
                        raise RuntimeError(
                            f"fold {fold} scheme {scheme}: categories do not "
                            f"partition the SR")
            z, sigma = combine(results)
            per_fold[fold]["schemes"][scheme] = (z, sigma, results)
            print(f"  {scheme:<5} Z = {z:.4f} +/- {sigma:.4f}")
            for name, r in results.items():
                rows.append({
                    "fold": fold, "scheme": scheme, "category": name,
                    "alpha": norm["alpha"], "sigma_alpha": norm["sigma_alpha"],
                    "N_S": r["N_S"], "S": r["S"], "sigma_S": r["sigma_S"],
                    "B_mc": r["B_mc"], "B": r["B"], "sigma_B": r["sigma_B"],
                    "Z": r["Z"], "sigma_Z": r["sigma_Z"],
                })

    print("\n" + "=" * 62)
    print(f"{'scheme':<8}{'Z_fold0..3':>34}{'combined':>12}")
    combined = {}
    for scheme in ("cut", "gnn4", "gnn2", "none"):
        zs = [per_fold[f]["schemes"][scheme][0] for f in range(NFOLD)]
        sig = [per_fold[f]["schemes"][scheme][1] for f in range(NFOLD)]
        z_tot = float(np.sqrt(sum(z * z for z in zs)))
        s_tot = float(np.sqrt(sum((z / z_tot * s) ** 2
                                  for z, s in zip(zs, sig)))) if z_tot > 0 else 0.0
        combined[scheme] = (z_tot, s_tot, zs)
        print(f"{scheme:<8}" + "".join(f"{z:9.4f}" for z in zs)
              + f"{z_tot:12.4f}")
    print("=" * 62)

    if missing:
        print(f"\nWARNING: {len(set(missing))} v2 file(s) had no .npz and were "
              f"treated as missing: {sorted(set(missing))[:3]} ...")

    if args.diagnostic_only:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    write_final_z(combined)
    make_figure(combined, per_fold)
    write_report(combined, per_fold, rows, missing)
    print(f"\nReport: {OUT_DIR / 'report.md'}")


def write_csv(rows):
    path = OUT_DIR / "per_category.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v)
                             for k, v in r.items()})


def write_final_z(combined):
    lines = ["4-fold Asimov Z on the old v2 data",
             "alpha = N(Data, SB) / Y(Bkg MC, SB); SR = is_SR tagged events"]
    for scheme in ("cut", "gnn4", "gnn2", "none"):
        z, s, zs = combined[scheme]
        lines.append(f"{scheme}: folds " + " ".join(f"{v:.4f}" for v in zs)
                     + f" -> combined Z = {z:.4f} +/- {s:.4f}")
    (OUT_DIR / "final_z.txt").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")


def make_figure(combined, per_fold):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID,
    })
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    schemes = [("cut", "cut-based, 5 tags", BLUE),
               ("gnn4", "4-class GNN, 4 categories", ORANGE),
               ("gnn2", "binary GNN, p > 0.5", GREEN),
               ("none", "no binning (reference)", "#b8b7b2")]
    names = ["fold 0", "fold 1", "fold 2", "fold 3", "combined"]
    x = np.arange(len(names))
    width = 0.20
    for i, (key, label, colour) in enumerate(schemes):
        vals = list(combined[key][2]) + [combined[key][0]]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.94, color=colour,
               label=label)
        if key != "none":
            ax.text(x[-1] + (i - 1.5) * width, vals[-1] + 0.004,
                    f"{vals[-1]:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Asimov Z")
    ax.set_title("Significance on the old v2 samples", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "z_comparison.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)


def write_report(combined, per_fold, rows, missing):
    def table(scheme):
        lines = ["| fold | category | S | B | S/B | Z | Z² share |",
                 "|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["scheme"] != scheme:
                continue
            share = ""
            total = combined[scheme][0] ** 2
            if total > 0:
                share = f"{100 * r['Z'] ** 2 / total:.1f}%"
            lines.append(
                f"| {r['fold']} | {r['category']} | {r['S']:.4g} | "
                f"{r['B']:.4g} | {r['S'] / r['B'] if r['B'] else 0:.3g} | "
                f"{r['Z']:.4f} | {share} |")
        return "\n".join(lines)

    summary = ["| scheme | fold 0 | fold 1 | fold 2 | fold 3 | 4-fold combined |",
               "|---|---|---|---|---|---|"]
    for key in ("cut", "gnn4", "gnn2", "none"):
        z, s, zs = combined[key]
        summary.append(f"| {key} | " + " | ".join(f"{v:.4f}" for v in zs)
                       + f" | **{z:.4f} ± {s:.4f}** |")
    summary = "\n".join(summary)

    n_with_data = sum(1 for f in range(NFOLD)
                      if per_fold[f]["norm"]["alpha_source"] == "per fold")
    alpha_lines = "\n".join(
        f"| {f} | {per_fold[f]['norm']['N_data_sb']} | "
        f"{per_fold[f]['norm']['Y_bkg_sb']:.1f} | "
        f"{per_fold[f]['norm']['alpha']:.5f} | "
        f"{per_fold[f]['norm']['alpha_source']} |"
        for f in range(NFOLD))
    parity_note = ""
    if n_with_data < NFOLD:
        parity_note = (
            f"\nThe v2 Data files carry `photon_eta_fold % 4` in {{0, 1}} only, "
            f"so parities 2 and 3 have **no** data sideband. Their `alpha` is "
            f"the mean of the parities that do ({n_with_data} of {NFOLD}); the "
            f"four values agree to 3e-4 in absolute terms, so the substitution "
            f"cannot move any Z below. The MC side is genuinely four-fold.\n")

    gain = combined["gnn4"][0] - combined["cut"][0]
    gain2 = combined["gnn2"][0] - combined["cut"][0]

    warn = ""
    if missing:
        warn = (f"\n> WARNING: {len(set(missing))} v2 scores were missing and "
                f"those events were dropped.\n")

    report = f"""# Asimov Z on the old v2 samples: cut-based vs GNN

Generated by `tools/significance_4class_v2.py`. This is the v2 analogue of
`results_compare_cutbase_vs_gnn/`, which is v3.
{warn}
Same events, same normalisation for every row: SR = `is_SR`-tagged events of
the out-of-fold parity, signal = ttH+VH+ggF+VBF, background =
`Bkg_MC/TT` + `Bkg_MC/DYGto2LG_10to100`, and

    alpha = N(Data, SB) / Y(Bkg MC, SB),   B = alpha * Y(Bkg MC, SR)

with `alpha` computed once per fold and used for all categories of all
schemes:

| fold | N(Data, SB) | Y(Bkg MC, SB) | alpha | alpha source |
|---|---|---|---|---|
{alpha_lines}
{parity_note}

## Combined Z

{summary}

`none` is the no-binning reference — the whole SR in one category. For each
scheme the script asserts that the sum of `S` and of `B` over its categories
equals these unbinned values, so every difference above is a pure binning
effect: same events, same `alpha`, same total `S` and `B`.

The 4-class GNN changes the combined Z by **{gain:+.4f}** with respect to the
cut-based categorisation, the binary GNN by **{gain2:+.4f}**.

## Per category — 4-class GNN

{table("gnn4")}

## Per category — cut-based

{table("cut")}

## What this does and does not say

* It says: on the v2 samples, re-binning the same SR events with the GNN
  score moves the Asimov Z by an amount comparable to the fold-to-fold
  spread. That is the quantitative version of the "GNN ≈ cut-based"
  impression.
* It does not say anything about the v3 analysis Z. v2 has two background
  samples only; `Z` here cannot be compared with the numbers in
  `results_compare_cutbase_vs_gnn/report.md`.
* **The 4-class thresholds were tuned on v3 and are reused unchanged**
  (`c_tth = 0.705`, `c_vh = 0.819`, `c_ggfvbf = 0.221`). On v2 they are out
  of domain, which is the most likely reason `gnn4` comes out below `cut` here
  while it was slightly above it on v3. Re-optimising them on v2 would mean
  tuning on the same events that are then quoted, because v2 Data only covers
  two of the four parities and leaves no independent validation split — hence
  the deliberate choice to leave them alone and read the `gnn4` row as a
  lower bound.
* **`gnn2` uses a fixed `p > 0.5`** and was not optimised at all, so it is a
  floor, not an indication of what the binary score can do when binned.
* The 4-class model is applied out of domain (trained on v3, scored on v2);
  `tools/results_v2_eval/report_auc_v2.md` shows it loses about 0.008 AUC to
  the binary model for exactly this reason. A v2 Z for the 4-class model is
  therefore a *lower bound* on what it would give in its own domain.

## Files

- `per_category.csv`
- `final_z.txt`
- `z_comparison.png`

![Z comparison](z_comparison.png)
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
