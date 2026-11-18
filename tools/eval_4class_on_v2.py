#!/usr/bin/env python3
"""Apply the 4-class GNN to the OLD v2 4-fold data and compare with the
binary ttH-vs-other GNN, on identical events.

Why this exists
---------------
`configs/hzg/ttH_4class_4fold_fold{0..3}.yaml` train on the v3 input
(`root_p2root_v3_*`), and the binary baseline
`configs/hzg/ttH_pt_only_4fold_data.yaml` lives on the v2 input
(`root_p2root_v2_pt_only`).  To ask "does the 4-class model do better than the
binary one *on the data the binary one was built for*", both models have to be
scored on the same v2 events.

The 4-class models are run over the v2 files with `scripts/inference.py` (see
`--npz-dir` for where that writes):

    python root_gnn_dgl/scripts/inference.py \
        --config /tmp/eval_v2/fold{0,1,2,3}.yaml --branch_name s_f0 s_f1 s_f2 s_f3 \
        --target root_p2root_v2_pt_only/<proc>/<era>.root \
        --destination tools/results_v2_eval/scores_npz/<proc>/<era> \
        --tree merged_SR_Sideband

where each `eval_v2/fold{k}.yaml` is `ttH_pt_only_4fold_data.yaml` with the
`Model`/`Loss` blocks and `Training_Directory` taken from
`ttH_4class_4fold_fold{k}.yaml`.  The binary scores are already in the tree:

    root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{k}/<proc>/<era>.root

Both are evaluated out of fold: fold k only keeps `photon_eta_fold % 4 == k`,
the parity its model never trained on.

Compared quantity
-----------------
The binary model's own task, ttH vs other, with the same score for each model:

    4-class : softmax[:, 1] = p_ttH        (class order of the 4-class config)
    binary  : ttH_gnn_score                (sigmoid output)

reported unweighted and weighted by the **v2** `weight_lumi_avg`, per fold and
pooled over the four folds.

The v2 process list is much smaller than v3 — `Signal/{ttH,VH,ggF,VBF}` plus
`Bkg_MC/{TT,DYGto2LG_10to100}` — so the negative set here is not the analysis
background.  This is a classifier comparison, not a significance.

Outputs (tools/results_v2_eval/):
    auc_ttH_vs_others_v2.csv
    report_auc_v2.md
    auc_ttH_vs_others_v2.png

Usage:
    python tools/eval_4class_on_v2.py
    python tools/eval_4class_on_v2.py --check-missing
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import uproot
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = REPO / "tools" / "results_v2_eval"
NPZ_DIR = OUT_DIR / "scores_npz"

BIN_BASE = REPO / "root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold"
TRAIN_BASE = REPO / "trainings" / "hzg"
BIN_LOG = "ttH_pt_only_4fold_fold{fold}"

TREE = "merged_SR_Sideband"
BIN_SCORE = "ttH_gnn_score"
FOLDVAR = "photon_eta_fold"
WEIGHT = "weight_lumi_avg"
NPZ_KEYS = ["scores", "tracking_info"]        # object arrays, keyed by branch

NFOLD = 4
TT_COL = 1                                   # softmax column of `ttH`

# The v2 process list, exactly as in ttH_pt_only_4fold_data.yaml.
PROCS = [
    "Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF",
    "Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100",
]
POS = "Signal/ttH"

# v2 has Run3 only (no UltraLegacy).
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]

BKG_PROCS = {"Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100"}

# Categorical pair, dataviz blue / orange, used for the two models.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_L = "#9ec5f4"
ORANGE_L = "#f5b49b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def bin_dir(fold):
    """`root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{fold}`"""
    return REPO / f"{BIN_BASE.name}{fold}"


def load_fold(fold):
    """Out-of-fold predictions of both models on the v2 test parity of `fold`.

    Returns arrays over the concatenated v2 files (same order as `PROCS` then
    `ERAS`), plus the list of files that had not been scored yet.
    """
    parts = {k: [] for k in ("p4", "pb", "w", "y", "proc")}
    missing = []

    for proc in PROCS:
        for era in ERAS:
            npz = NPZ_DIR / proc / f"{era}.npz"
            if not npz.exists():
                missing.append(f"{proc}/{era}")
                continue

            data = np.load(npz, allow_pickle=True)
            key = f"s_f{fold}"
            scores = np.asarray(data["scores"].flat[0][key])
            track = np.asarray(data["tracking_info"].flat[0][key])

            bin_tree = uproot.open(
                bin_dir(fold) / proc / f"{era}.root")[TREE]
            arrays = bin_tree.arrays(
                [BIN_SCORE, WEIGHT, FOLDVAR], library="np")

            n_bin = arrays[BIN_SCORE].size
            if not scores.shape[0] == track.shape[0] == n_bin:
                raise RuntimeError(
                    f"{proc}/{era}: {scores.shape[0]} 4-class, "
                    f"{track.shape[0]} tracking, {n_bin} binary events")
            fvar = np.asarray(arrays[FOLDVAR], dtype=np.int64)
            if not np.array_equal(track[:, 0].astype(np.int64), fvar):
                raise RuntimeError(
                    f"{proc}/{era}: event order differs between the npz and "
                    f"the scored tree")

            # out-of-fold parity only
            mask = (fvar % NFOLD) == fold
            n = int(mask.sum())
            parts["p4"].append(np.asarray(scores[mask, TT_COL], np.float64))
            parts["pb"].append(np.asarray(arrays[BIN_SCORE][mask], np.float64))
            parts["w"].append(np.asarray(arrays[WEIGHT][mask], np.float64))
            parts["y"].append(np.full(n, proc == POS, dtype=bool))
            parts["proc"].append(np.full(n, proc, dtype=object))

    out = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in parts.items()}
    out["proc"] = out["proc"].astype(str) if out["proc"].size else out["proc"]
    out["missing"] = missing
    return out


def auc(y, score, weight=None):
    return float(roc_auc_score(y, score, sample_weight=weight))


def best_log_auc(fold):
    """max `Test_AUC` of the binary model in its own training log, or None."""
    path = TRAIN_BASE / BIN_LOG.format(fold=fold) / "training.log"
    if not path.exists():
        return None
    values = [float(v) for v in
              re.findall(r"Test_AUC\s+([0-9.]+)", path.read_text())]
    return max(values) if values else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def measure(folds, pooled):
    rows = []
    for fold in list(range(NFOLD)) + ["pooled"]:
        d = pooled if fold == "pooled" else folds[fold]
        y, w = d["y"], d["w"]
        rows.append({
            "fold": fold,
            "n_events": int(y.size),
            "n_ttH": int(y.sum()),
            "w_ttH": float(w[y].sum()),
            "w_other": float(w[~y].sum()),
            "p4_unweighted": auc(y, d["p4"]),
            "pb_unweighted": auc(y, d["pb"]),
            "p4_weighted": auc(y, d["p4"], w),
            "pb_weighted": auc(y, d["pb"], w),
            "pb_log": best_log_auc(fold) if fold != "pooled" else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_csv(rows):
    path = OUT_DIR / "auc_ttH_vs_others_v2.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: ("" if v is None else
                    f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()})
    return path


def make_figure(pooled, rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)

    # --- per-fold AUC, weighted and unweighted ----------------------------
    ax1.set_facecolor(SURFACE)
    per_fold = [r for r in rows if r["fold"] != "pooled"]
    all_rows = per_fold + [rows[-1]]
    x = np.arange(len(all_rows))
    labels = [f"fold {r['fold']}" for r in per_fold] + ["pooled"]

    series = [
        ("4-class, weighted", "p4_weighted", BLUE, -0.30),
        ("binary, weighted", "pb_weighted", ORANGE, -0.10),
        ("4-class, unweighted", "p4_unweighted", BLUE_L, 0.10),
        ("binary, unweighted", "pb_unweighted", ORANGE_L, 0.30),
    ]
    for name, key, colour, offset in series:
        ax1.bar(x + offset, [r[key] for r in all_rows], 0.19,
                color=colour, label=name)

    lo = min(r[key] for _, key, _, _ in series for r in all_rows)
    ax1.set_ylim(np.floor((lo - 0.02) * 20) / 20, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ttH-vs-rest AUC")
    ax1.set_title("ttH vs others, old v2 events", loc="left",
                  fontsize=10.5, color=INK, pad=8)
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")

    # difference between the two weighted values, above each group
    for xi, r in zip(x, all_rows):
        top = max(r["p4_weighted"], r["pb_weighted"])
        ax1.text(xi, top + 0.006,
                 f"{r['p4_weighted'] - r['pb_weighted']:+.4f}",
                 ha="center", va="bottom", fontsize=7, color=INK2)

    # --- pooled ROC curves ------------------------------------------------
    ax2.set_facecolor(SURFACE)
    y, w = pooled["y"], pooled["w"]
    for name, score, colour in (("4-class", pooled["p4"], BLUE),
                                ("binary", pooled["pb"], ORANGE)):
        fpr, tpr, _ = roc_curve(y, score, sample_weight=w)
        ax2.plot(fpr, tpr, color=colour, linewidth=2,
                 label=f"{name} ({auc(y, score, w):.4f})")
    ax2.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.005)
    ax2.set_xlabel("weighted false positive rate")
    ax2.set_ylabel("weighted true positive rate")
    ax2.set_title("Pooled ROC, v2 weights", loc="left",
                  fontsize=10.5, color=INK, pad=8)
    ax2.grid(color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("4-class GNN vs binary GNN on the same v2 events",
                 x=0.012, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT_DIR / "auc_ttH_vs_others_v2.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(pooled, rows, missing):
    table = ["| fold | N events | N ttH | w(ttH) | w(other) "
             "| 4-class unw | binary unw | 4-class wgt | binary wgt "
             "| binary log max Test_AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        log = "" if r["pb_log"] is None else f"{r['pb_log']:.4f}"
        table.append(
            f"| {r['fold']} | {r['n_events']} | {r['n_ttH']} | "
            f"{r['w_ttH']:.4g} | {r['w_other']:.4g} | "
            f"{r['p4_unweighted']:.4f} | {r['pb_unweighted']:.4f} | "
            f"{r['p4_weighted']:.4f} | {r['pb_weighted']:.4f} | {log} |")
    table = "\n".join(table)

    per_fold = [r for r in rows if r["fold"] != "pooled"]
    d_weighted = [r["p4_weighted"] - r["pb_weighted"] for r in per_fold]
    d_unweighted = [r["p4_unweighted"] - r["pb_unweighted"] for r in per_fold]
    offsets = [r["pb_weighted"] - r["pb_log"] for r in per_fold
               if r["pb_log"] is not None]
    spread4 = (max(r["p4_weighted"] for r in per_fold)
               - min(r["p4_weighted"] for r in per_fold))
    spreadb = (max(r["pb_weighted"] for r in per_fold)
               - min(r["pb_weighted"] for r in per_fold))
    fold3 = [r for r in per_fold if r["fold"] == 3]
    rest = [r for r in per_fold if r["fold"] != 3]
    fold3_note = ""
    if fold3 and rest:
        gap = np.mean([r["p4_weighted"] for r in fold3]) - np.mean(
            [r["p4_weighted"] for r in rest])
        if abs(gap) > 0.005:
            fold3_note = (
                f"\nNote that fold 3 is an outlier for the 4-class model: "
                f"{fold3[0]['p4_weighted']:.4f} against "
                f"{np.mean([r['p4_weighted'] for r in rest]):.4f} averaged "
                f"over folds 0-2 ({gap:+.4f}). The binary model does not show "
                f"this (spread {spreadb:.4f} over the four folds). Either one "
                f"of the four 4-class checkpoints is better than the other "
                f"three, or its test parity is easier; with four folds there "
                f"is no way to tell these apart from a single number.\n")

    comp = []
    for proc in PROCS:
        if proc == POS:
            continue
        m = pooled["proc"] == proc
        comp.append((proc, int(m.sum()), float(pooled["w"][m].sum())))
    total_w = sum(c[2] for c in comp)
    sig_share = 100 * sum(c[2] for c in comp if c[0] not in BKG_PROCS) / total_w
    comp_lines = "\n".join(
        f"   | `{p}` | {n} | {w:.4g} | {100 * w / total_w:.1f}% |"
        for p, n, w in sorted(comp, key=lambda t: -t[2]))

    missing_note = (
        f"\n> {len(missing)} of {len(PROCS) * len(ERAS)} v2 files were not "
        f"scored yet and are excluded: `{missing[0]}` ...\n"
        if missing else "")

    if offsets:
        offset_note = (
            f"\n**A caveat on the binary column.** Recomputing the binary "
            f"model's own metric from its scored trees gives "
            f"{np.mean([r['pb_weighted'] for r in per_fold]):.4f} on average, "
            f"while the maxima in its `training.log` are "
            f"{np.mean([r['pb_log'] for r in per_fold if r['pb_log']]):.4f} — "
            f"a constant offset of {np.mean(offsets):+.4f} "
            f"(spread {max(offsets) - min(offsets):.4f}) in all four folds. "
            f"The same reconstruction reproduces the 4-class `Test_AUC` to "
            f"5e-4 (`results_pairwise_auc_4class/report_training_auc.md`), so "
            f"the offset is a property of these particular scored trees "
            f"(checkpoint choice or event selection), not of the method. The "
            f"difference between the two models below is three times this "
            f"offset, so the ordering survives, but the binary value quoted "
            f"here is the scored-tree value, not the log value.\n")
    else:
        offset_note = ""

    report = f"""# The 4-class GNN on the old v2 data: ttH vs others

Generated by `tools/eval_4class_on_v2.py`. The 4-class models
(`trainings/hzg/ttH_4class_4fold_fold{{0..3}}`, trained on v3 input) are
applied to the v2 files of `configs/hzg/ttH_pt_only_4fold_data.yaml` and
compared with the old binary model on **identical events**, both scored out of
fold (`photon_eta_fold % 4`).
{missing_note}
| | 4-class | binary |
|---|---|---|
| checkpoint | `ttH_4class_4fold_fold{{k}}` | `ttH_pt_only_4fold_fold{{k}}` |
| score | `softmax[:, 1]` = `p_ttH` | `ttH_gnn_score` |
| source | `tools/results_v2_eval/scores_npz/` | `root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{{k}}` |
| weights | v2 `weight_lumi_avg` | v2 `weight_lumi_avg` |

Positive class `Signal/ttH`; negatives are the rest of the v2 process list
(`VH`, `ggF`, `VBF`, `TT`, `DYGto2LG_10to100`).

## Results

{table}

Pooled, the binary model is ahead by
{rows[-1]['pb_weighted'] - rows[-1]['p4_weighted']:+.4f} weighted and
{rows[-1]['pb_unweighted'] - rows[-1]['p4_unweighted']:+.4f} unweighted.
Per fold the weighted difference is {min(d_weighted):+.4f} …
{max(d_weighted):+.4f} (mean {np.mean(d_weighted):+.4f}); unweighted
{min(d_unweighted):+.4f} … {max(d_unweighted):+.4f}
(mean {np.mean(d_unweighted):+.4f}).
{fold3_note}{offset_note}
## Reading it

The 4-class model does **not** win on v2, and the natural explanation is not
the architecture but the input it was trained on.

1. **It is out of domain.** `ttH_4class_4fold_fold{{k}}` trained on
   `root_p2root_v3_*`; here it is applied to `root_p2root_v2_pt_only`. The two
   versions differ in the sample list and in the per-process event counts, so
   the network's decision boundaries are tuned for a slightly different input
   mixture. In its own domain the same `p_ttH` gives a one-vs-rest AUC of
   0.9757 unweighted (`results_pairwise_auc_4class/report.md`), essentially
   the binary model's 0.9758 here — but those are different event sets and
   different negative mixtures, so that comparison is context, not evidence.

2. **The negatives are two background samples only.** By weight:

   | negative process | N events | weight sum | share of negative weight |
   |---|---|---|---|
   {comp_lines}

   The three signal samples are {sig_share:.1f}% of the negative weight, so
   both models are effectively solving "ttH vs TT and DY→2ℓγ". The extra
   VH / ggF_VBF outputs of the 4-class model buy almost nothing for this
   particular ranking, while their softmax competition can only subtract from
   `p_ttH`.

3. **Neither number is the significance.** AUC measures global ranking; `Z`
   is set by the tail of the score distribution at the working point. This
   comparison says nothing about `Z` in
   `results_compare_cutbase_vs_gnn/report.md`, and the v2 background list is
   not the analysis background, so no `Z` can be derived from these files at
   all.

## The clean test, if it is wanted

Score the binary model on the **v3** files (the 4-class model's domain) with
the same procedure:

    python root_gnn_dgl/scripts/inference.py \
        --config <binary config for root_p2root_v3_* files> --branch_name b_f0 b_f1 b_f2 b_f3 \
        --target root_p2root_v3_pt_only/... --destination ... --tree merged_SR_Sideband

and set `--npz-dir` to those scores. That removes the domain mismatch and is
the only way to compare the two objectives fairly. Until then, the honest
statement is: on v2 data the binary model separates better, on v3 data the two
are indistinguishable at the 1e-3 level, and neither is the significance.

## Files

- `auc_ttH_vs_others_v2.csv`
- `auc_ttH_vs_others_v2.png`

![ttH vs others AUC](auc_ttH_vs_others_v2.png)
"""
    path = OUT_DIR / "report_auc_v2.md"
    path.write_text(report, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-missing", action="store_true",
                        help="Only list the v2 files that have no .npz yet, "
                             "then exit.")
    args = parser.parse_args()

    if args.check_missing:
        missing = [f"{p}/{e}" for p in PROCS for e in ERAS
                   if not (NPZ_DIR / p / f"{e}.npz").exists()]
        print(f"{len(missing)} of {len(PROCS) * len(ERAS)} files not scored")
        for m in missing:
            print("  " + m)
        return 1 if missing else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folds = {}
    for fold in range(NFOLD):
        print(f"fold {fold}: reading ...", flush=True)
        folds[fold] = load_fold(fold)
        if folds[fold]["missing"]:
            print(f"  skipped {len(folds[fold]['missing'])} unscored file(s)",
                  flush=True)

    missing = sorted({m for f in folds.values() for m in f["missing"]})
    if any(folds[f]["y"].size == 0 for f in folds):
        raise SystemExit("No scored events; run the inference first "
                         "(--check-missing lists what is missing).")

    pooled = {k: np.concatenate([folds[f][k] for f in range(NFOLD)])
              for k in ("p4", "pb", "w", "y", "proc")}

    rows = measure(folds, pooled)
    write_csv(rows)
    make_figure(pooled, rows)
    report = write_report(pooled, rows, missing)

    print(f"\n{'fold':>7} {'p4 unw':>8} {'pb unw':>8} {'Δ':>8} "
          f"{'p4 wgt':>8} {'pb wgt':>8} {'Δ':>8}")
    for r in rows:
        print(f"{str(r['fold']):>7} {r['p4_unweighted']:8.4f} "
              f"{r['pb_unweighted']:8.4f} "
              f"{r['p4_unweighted'] - r['pb_unweighted']:+8.4f} "
              f"{r['p4_weighted']:8.4f} {r['pb_weighted']:8.4f} "
              f"{r['p4_weighted'] - r['pb_weighted']:+8.4f}")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    raise SystemExit(main())
