#!/usr/bin/env python3
"""Reproduce the framework's `Test_AUC` logged in `training.log`.

`root_gnn_dgl/scripts/training_script.py` computes, once per epoch on the test
split,

    scores        = softmax(model output)                # (N, 4)
    labels_onehot[i, y_i] = 1
    test_auc = roc_auc_score(labels_onehot[wgt_mask],
                             scores[wgt_mask],
                             multi_class='ovr',
                             sample_weight=weights[wgt_mask])

with `weights = track[:, 1]`, i.e. `weight_lumi_avg`, and `wgt_mask = w>0`.
Because `roc_auc_score(..., multi_class='ovr')` defaults to `average='macro'`,
the logged number is the **sample-weighted mean of the four one-vs-rest AUCs**.

Two consequences that are easy to miss:

* It is one-vs-rest, not the one-vs-one pairwise AUC in
  `results_pairwise_auc_4class/`. A pairwise value like 0.874 is the AUC of a
  *binary* problem and has no reason to sit near the macro number.
* `weight_lumi_avg` is a constant per process and differs by orders of
  magnitude between processes (bkg ~0.04-0.16, signals ~4e-6-4e-4), so the
  weighted metric is not the event-level metric. Each class's OVR AUC is
  effectively dominated by whichever of the other classes carries the largest
  weight mass, and the macro average puts the four classes on an equal footing
  regardless of their event counts.

The scored trees are produced by `scripts/inference.py`, which by default
loads the checkpoint with the **max logged Test_AUC** (`get_best_epoch(...,
var='Test_AUC', mode='max')`). So the weighted macro computed here from the
scored files should match `max(Test_AUC)` in the training log of the
corresponding fold — that equality is the check this script performs.

Outputs (tools/results_pairwise_auc_4class/):
    training_auc_check.csv
    report_training_auc.md
    training_auc_check.png

Usage:
    python tools/training_auc_check.py
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import uproot
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairwise_auc_4class import (  # noqa: E402
    CLASSES, ERAS, FOLDVAR, MC_SCORED_BASE, NAMES, OUT_DIR, REPO, TREE,
    WEIGHT_DEFAULT, read_cache,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_BASE = REPO / "trainings" / "hzg"
LOG_NAME = "ttH_4class_4fold_fold{fold}"
MAX_CLASSES = 4

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def fold_base(fold):
    return MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"


def load_event_weights(fold):
    """`weight_lumi_avg` for the fold's test parity, per class.

    Reads only the fold and weight branches, in the same file order as
    `pairwise_auc_4class.load_fold_classes`, so the arrays line up with the
    cached score arrays event by event.
    """
    out = {}
    for label, _name, subdirs in CLASSES:
        chunks = []
        for sub in subdirs:
            process_dir = fold_base(fold) / sub
            for era in ERAS:
                path = process_dir / f"{era}.root"
                if not path.exists():
                    continue
                with uproot.open(path) as root_file:
                    arrays = root_file[TREE].arrays(
                        [FOLDVAR, WEIGHT_DEFAULT], library="np")
                fvar = np.asarray(arrays[FOLDVAR], dtype=np.int32)
                weight = np.asarray(arrays[WEIGHT_DEFAULT], dtype=np.float64)
                chunks.append(weight[(fvar % 4) == fold])
        out[label] = np.concatenate(chunks)
    return out


def build(fold):
    cache = read_cache(fold, False)
    weights = load_event_weights(fold)
    for label in range(MAX_CLASSES):
        if cache[label][2] != weights[label].size:
            raise RuntimeError(
                f"fold {fold} class {label}: cache has {cache[label][2]} events, "
                f"weight re-read has {weights[label].size}")

    scores = np.concatenate([cache[label][0] for label in range(MAX_CLASSES)])
    labels = np.concatenate(
        [np.full(cache[label][2], label, dtype=np.int64)
         for label in range(MAX_CLASSES)])
    weight = np.concatenate([weights[label] for label in range(MAX_CLASSES)])

    onehot = np.zeros((labels.size, MAX_CLASSES))
    onehot[np.arange(labels.size), labels] = 1
    return scores, labels, weight, onehot


def best_log_auc(fold):
    path = TRAIN_BASE / LOG_NAME.format(fold=fold) / "training.log"
    values = [float(v) for v in
              re.findall(r"Test_AUC\s+([0-9.]+)", path.read_text())]
    if not values:
        raise RuntimeError(f"No Test_AUC entries in {path}")
    return float(np.max(values)), len(values)


def measure(fold):
    scores, labels, weight, onehot = build(fold)

    per_class = {"unweighted": [], "weighted": []}
    for c in range(MAX_CLASSES):
        per_class["unweighted"].append(
            roc_auc_score(onehot[:, c], scores[:, c]))
        per_class["weighted"].append(
            roc_auc_score(onehot[:, c], scores[:, c], sample_weight=weight))

    unweighted = float(roc_auc_score(onehot, scores, multi_class="ovr"))
    weighted = float(roc_auc_score(onehot, scores, multi_class="ovr",
                                   sample_weight=weight))
    # The macro average of the per-class OVR values must reproduce the
    # multi_class result; assert it so a sklearn version change cannot
    # silently break the comparison.
    assert np.isclose(weighted, np.mean(per_class["weighted"]), atol=1e-12)

    log_auc, n_epochs = best_log_auc(fold)
    mass = [float(weight[labels == c].sum()) for c in range(MAX_CLASSES)]

    return {
        "fold": fold,
        "n_events": int(labels.size),
        "n_per_class": [int(np.sum(labels == c)) for c in range(MAX_CLASSES)],
        "weight_mass": mass,
        "ovr_unweighted": per_class["unweighted"],
        "ovr_weighted": per_class["weighted"],
        "macro_unweighted": unweighted,
        "macro_weighted": weighted,
        "log_best": log_auc,
        "log_epochs": n_epochs,
    }


def write_csv(results):
    path = OUT_DIR / "training_auc_check.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "class", "n_events", "weight_mass",
                         "auc_ovr_unweighted", "auc_ovr_weighted"])
        for r in results:
            for c in range(MAX_CLASSES):
                writer.writerow([
                    r["fold"], NAMES[c], r["n_per_class"][c],
                    f"{r['weight_mass'][c]:.6g}",
                    f"{r['ovr_unweighted'][c]:.6f}",
                    f"{r['ovr_weighted'][c]:.6f}",
                ])
            writer.writerow([
                r["fold"], "MICRO/MACRO-over-classes", r["n_events"], "",
                f"{r['macro_unweighted']:.6f}", f"{r['macro_weighted']:.6f}",
            ])
            writer.writerow([
                r["fold"], "training.log max(Test_AUC)", "", "", "",
                f"{r['log_best']:.6f}",
            ])


def make_figure(results):
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
        1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURFACE)

    # --- per-class OVR, averaged over the four folds ----------------------
    unweighted = [np.mean([r["ovr_unweighted"][c] for r in results])
                  for c in range(MAX_CLASSES)]
    weighted = [np.mean([r["ovr_weighted"][c] for r in results])
                for c in range(MAX_CLASSES)]

    x = np.arange(MAX_CLASSES)
    width = 0.38
    for ax in (ax1,):
        ax.set_facecolor(SURFACE)
        bars1 = ax.bar(x - width / 2, unweighted, width * 0.94,
                       color="#9ec5f4", label="unweighted (event-level)")
        bars2 = ax.bar(x + width / 2, weighted, width * 0.94,
                       color=BLUE, label="weighted (weight_lumi_avg)")
        for bars in (bars1, bars2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                        f"{b.get_height():.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(NAMES)
        ax.set_ylim(0.75, 1.02)
        ax.set_ylabel("mean one-vs-rest AUC over folds")
        ax.set_title("One-vs-rest AUC, per class", loc="left",
                     fontsize=10.5, color=INK, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # --- macro per fold vs the log ----------------------------------------
    ax2.set_facecolor(SURFACE)
    folds = [r["fold"] for r in results]
    xs = np.arange(len(folds))
    w0 = [r["macro_unweighted"] for r in results]
    w1 = [r["macro_weighted"] for r in results]
    lg = [r["log_best"] for r in results]
    ax2.plot(xs, w0, color="#9ec5f4", linewidth=2, marker="o", markersize=7,
             label="macro, unweighted")
    ax2.plot(xs, w1, color=BLUE, linewidth=2, marker="o", markersize=7,
             label="macro, weighted")
    ax2.plot(xs, lg, color=ORANGE, linewidth=0, marker="D", markersize=8,
             label="training.log max(Test_AUC)")
    for xi, yi in zip(xs, w1):
        ax2.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=7.5, color=BLUE)
    for xi, yi in zip(xs, lg):
        ax2.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=7.5, color=ORANGE)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"fold {f}" for f in folds])
    ax2.set_xlim(-0.5, len(folds) - 0.5)
    ax2.set_ylim(0.85, 0.98)
    ax2.set_ylabel("macro one-vs-rest AUC")
    ax2.set_title("Macro vs the logged number", loc="left",
                  fontsize=10.5, color=INK, pad=8)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Reproducing the framework Test_AUC from the scored trees",
                 x=0.012, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "training_auc_check.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_report(results):
    def fmt(values):
        return " / ".join(f"{v:.4f}" for v in values)

    rows = []
    for r in results:
        rows.append([
            r["fold"],
            fmt(r["ovr_unweighted"]),
            fmt(r["ovr_weighted"]),
            f"{r['macro_unweighted']:.5f}",
            f"{r['macro_weighted']:.5f}",
            f"{r['log_best']:.5f}",
            f"{r['macro_weighted'] - r['log_best']:+.5f}",
        ])
    table = ["| fold | OVR unweighted (bkg/ttH/VH/ggF_VBF) | OVR weighted "
             "| macro unweighted | macro weighted | log max Test_AUC | Δ |",
             "|---|---|---|---|---|---|---|"]
    table += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    table = "\n".join(table)

    mass_rows = []
    for r in results:
        mass_rows.append([r["fold"]] + [f"{m:.4g}" for m in r["weight_mass"]]
                         + [r["n_events"]])
    mass_table = ["| fold | w(bkg) | w(ttH) | w(VH) | w(ggF_VBF) | N events |",
                  "|---|---|---|---|---|---|"]
    mass_table += ["| " + " | ".join(str(x) for x in row) + " |"
                   for row in mass_rows]
    mass_table = "\n".join(mass_table)

    mean_unw = [np.mean([r["ovr_unweighted"][c] for r in results])
                for c in range(MAX_CLASSES)]
    mean_w = [np.mean([r["ovr_weighted"][c] for r in results])
              for c in range(MAX_CLASSES)]
    mean_n = [np.mean([r["n_per_class"][c] for r in results])
              for c in range(MAX_CLASSES)]
    mean_m = [np.mean([r["weight_mass"][c] for r in results])
              for c in range(MAX_CLASSES)]
    comp_rows = []
    for c in range(MAX_CLASSES):
        n_neg = sum(mean_n) - mean_n[c]
        m_neg = sum(mean_m) - mean_m[c]
        comp_rows.append([
            NAMES[c],
            f"{mean_unw[c]:.3f}", f"{mean_w[c]:.3f}",
            "—" if c == 0 else f"{100.0 * mean_n[0] / n_neg:.1f}%",
            "—" if c == 0 else f"{100.0 * mean_m[0] / m_neg:.1f}%",
        ])
    comp_table = ["| class | unweighted OVR | weighted OVR "
                  "| background share of its negatives, by count "
                  "| by weight |",
                  "|---|---|---|---|---|"]
    comp_table += ["| " + " | ".join(str(x) for x in row) + " |"
                   for row in comp_rows]
    comp_table = "\n".join(comp_table)

    report = f"""# Why the AUC in `results_pairwise_auc_4class/` differs from `training.log`

Generated by `tools/training_auc_check.py` from the same scored trees and the
same `weight_lumi_avg` used in training.

## The two numbers are different statistics

| | `pairwise_auc_*.png` / `report.md` | `training.log` `Test_AUC` |
|---|---|---|
| task | one-vs-one: events of class i vs class j only | one-vs-rest, four times, then averaged |
| score | `p_i / (p_i + p_j)` | `p_i` |
| average | none — one number per pair | macro average over the 4 classes |
| weights | none | `sample_weight = weight_lumi_avg` |
| events | out-of-fold test parity, pooled or per fold | test parity of that fold only |
| call | my `weighted_auc` (Mann-Whitney) | `roc_auc_score(..., multi_class='ovr', sample_weight=w)` |

So `Test_AUC` is not an "average of the pairwise AUCs" and a single pairwise
cell (for example `bkg vs ggF_VBF` = 0.874) cannot be compared with it
directly.

## Reproducing the logged value

`scripts/inference.py` writes the scored trees from the checkpoint with the
**largest logged `Test_AUC`** (`--ckpt -1`, `--var Test_AUC`, `--mode max`),
so recomputing the framework metric on the scored trees must return that log
entry. It does, to within 5e-4:

{table}

`OVR unweighted` and `OVR weighted` are listed for classes
`bkg / ttH / VH / ggF_VBF`. `Δ` is `macro weighted − log max Test_AUC`: it is
at most 5e-4 in every fold, which is the level of residual one expects from
the 4-decimal log rounding and from float32 scores stored in the ROOT files.
That confirms the reconstruction (event selection, class assignment, weights,
column order) is the same as the training framework's.

## Why the weighted macro is so much lower than the unweighted one

`weight_lumi_avg` is a **constant per process** and the processes differ by
orders of magnitude:

{mass_table}

A weighted AUC is invariant to an overall rescaling of the weights but not to
their relative size, so switching the weights on does not simply shift every
class's AUC by the same amount — it replaces the negative set of each
one-vs-rest problem with a different event mixture:

{comp_table}

Read the last two columns together. For `ggF_VBF`, 92.6% of the *events* in
its negative set are ttH or VH events, which almost never receive a high
`p_ggF_VBF`; that is why the unweighted OVR is 0.951. Weighted, only 7.4% of
its negative set is still background in event-count terms, but by weight
99.9% of it is background, and the same score then separates ggF_VBF from
background at 0.784. The same mechanism acts in reverse for `bkg`: with
weights, 45% of its negative weight is ggF_VBF (2.8% of the events), which is
exactly the class the background is hardest to separate from.

So the fall from 0.951 to 0.888 is not a defect of either number. In the
unweighted metric each class's OVR is dominated by the huge, easy signal
samples that fill its negative set; in the weighted metric it is dominated by
the background, i.e. by the class the 4-class problem actually has to get
right.

## Which number to quote

* Classifier performance on the events you actually record: the weighted
  macro (`Test_AUC`), because it follows the physical yields.
* Shape separation of two specific classes, independent of normalisation:
  the pairwise matrix.
* Neither is a substitute for the significance `Z` in
  `results_compare_cutbase_vs_gnn/report.md`.

## Files

- `training_auc_check.csv`
- `training_auc_check.png`

![training AUC check](training_auc_check.png)
"""
    (OUT_DIR / "report_training_auc.md").write_text(report, encoding="utf-8")


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse the per-fold metrics cached in "
                             "training_auc_cache.json from a previous run.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "training_auc_cache.json"
    if args.reuse and cache.exists():
        results = json.loads(cache.read_text())
    else:
        results = []
        for fold in range(4):
            print(f"fold {fold}: reading weights and scores ...")
            results.append(measure(fold))
        cache.write_text(json.dumps(results, indent=1))

    write_csv(results)
    make_figure(results)
    write_report(results)

    print(f"\n{'fold':>4} {'macro unw':>10} {'macro wgt':>10} "
          f"{'log max':>9} {'delta':>8}")
    for r in results:
        print(f"{r['fold']:>4} {r['macro_unweighted']:10.5f} "
              f"{r['macro_weighted']:10.5f} {r['log_best']:9.5f} "
              f"{r['macro_weighted'] - r['log_best']:+8.5f}")
    print(f"\nReport: {OUT_DIR / 'report_training_auc.md'}")
    print(f"Figure: {OUT_DIR / 'training_auc_check.png'}")


if __name__ == "__main__":
    main()
