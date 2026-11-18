#!/usr/bin/env python3
"""Pairwise (one-vs-one) ROC AUC for the 4-class H->Zgamma GNN.

The 4-class model predicts a softmax vector

    ttH_gnn_score = [p_bkg, p_ttH, p_VH, p_ggF_VBF]

(columns 0-3 = config labels 0-3).  The scored ROOT files do not contain a
label branch, so the true class is taken from the process directory, exactly
as in the training config:

    label 0  bkg      Bkg_MC/{TT, DYGto2LG_*, DYJetsToLL_Stitch}
    label 1  ttH      Signal/ttH
    label 2  VH       Signal/VH
    label 3  ggF_VBF  Signal/ggF + Signal/VBF

Pairwise AUC for classes (i, j)
------------------------------
Restrict to events whose true class is i or j and rank by the conditional
pair probability

    score = p_i / (p_i + p_j)

with class i as positive.  AUC > 0.5 means the model separates i from j in
the correct direction.  `--raw-p` uses p_i alone instead.

Only the out-of-fold test parity of each fold directory is used:
fold k uses `photon_eta_fold % 4 == k`, which is the parity its model never
trained on.  Combining the four folds therefore pools all events, each
scored by the model that did not train on it.

Outputs (tools/results_pairwise_auc_4class/):
    pairwise_auc_per_fold.csv
    pairwise_auc_pooled.csv
    ovr_auc_pooled.csv
    auc_matrix.csv
    report.md
    pairwise_auc_fold0.png ... pairwise_auc_fold3.png
    pairwise_auc_4folds.png
    pairwise_auc_matrix.png

Usage:
    python tools/pairwise_auc_4class.py
    python tools/pairwise_auc_4class.py --weighted
    python tools/pairwise_auc_4class.py --no-cache   # re-read ROOT files
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
OUT_DIR = REPO / "tools" / "results_pairwise_auc_4class"
CACHE_DIR = OUT_DIR / "cache"

MC_SCORED_BASE = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"

TREE = "merged_SR_Sideband"
SCORE = "ttH_gnn_score"
FOLDVAR = "photon_eta_fold"
WEIGHT_DEFAULT = "weight_lumi_avg"

ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

# label -> (display name, list of process directories)
CLASSES = [
    (0, "bkg", [
        "Bkg_MC/TT",
        "Bkg_MC/DYGto2LG_10to100",
        "Bkg_MC/DYGto2LG_100to200",
        "Bkg_MC/DYGto2LG_200to400",
        "Bkg_MC/DYGto2LG_400to600",
        "Bkg_MC/DYGto2LG_600",
        "Bkg_MC/DYJetsToLL_Stitch",
    ]),
    (1, "ttH", ["Signal/ttH"]),
    (2, "VH", ["Signal/VH"]),
    (3, "ggF_VBF", ["Signal/ggF", "Signal/VBF"]),
]

NAMES = [c[1] for c in CLASSES]

# Validated single-hue sequential ramp (dataviz blue steps 100 -> 600).
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _stack_scores(sc):
    sc = np.asarray(sc)
    if sc.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if sc.dtype == object or sc.ndim == 1:
        sc = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in sc])
    sc = np.asarray(sc, dtype=np.float32)
    if sc.ndim != 2 or sc.shape[1] != 4:
        raise ValueError(f"{SCORE} must be (N,4), got {sc.shape}")
    return sc


def load_dir(base, subdir, with_weight):
    process_dir = base / subdir
    if not process_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {process_dir}")

    branches = [SCORE, FOLDVAR]
    if with_weight:
        branches.append(WEIGHT_DEFAULT)

    scores, folds, weights = [], [], []
    for era in ERAS:
        path = process_dir / f"{era}.root"
        if not path.exists():
            continue
        with uproot.open(path) as root_file:
            if TREE not in root_file:
                raise KeyError(f"{TREE!r} not found in {path}")
            tree = root_file[TREE]
            missing = [b for b in branches if b not in tree.keys()]
            if missing:
                raise KeyError(f"Missing branches {missing} in {path}")
            arrays = tree.arrays(branches, library="np")

        scores.append(_stack_scores(arrays[SCORE]))
        folds.append(np.asarray(arrays[FOLDVAR], dtype=np.int32))
        n = scores[-1].shape[0]
        if with_weight:
            weights.append(np.asarray(arrays[WEIGHT_DEFAULT], dtype=np.float64))
        else:
            weights.append(np.ones(n, dtype=np.float64))

    if not scores:
        raise RuntimeError(f"No ROOT files loaded for {subdir} under {base}")
    return (np.concatenate(scores), np.concatenate(folds),
            np.concatenate(weights))


def load_fold_classes(fold, with_weight):
    base = MC_SCORED_BASE.parent / f"{MC_SCORED_BASE.name}{fold}"
    data = {}
    for label, name, subdirs in CLASSES:
        parts = [load_dir(base, sub, with_weight) for sub in subdirs]
        score = np.concatenate([p[0] for p in parts])
        fvar = np.concatenate([p[1] for p in parts])
        weight = np.concatenate([p[2] for p in parts])
        mask = (fvar % 4) == fold
        data[label] = (score[mask], weight[mask], int(mask.sum()))
    return data


# ---------------------------------------------------------------------------
# Optional cache: reading ~13 GB of scored trees takes several minutes
# ---------------------------------------------------------------------------

def cache_path(fold, with_weight):
    return CACHE_DIR / f"fold{fold}_w{int(with_weight)}.npz"


def save_cache(fold, data, with_weight):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for label in range(4):
        score, weight, count = data[label]
        arrays[f"s{label}"] = score
        arrays[f"w{label}"] = weight
        arrays[f"n{label}"] = np.array(count)
    np.savez(cache_path(fold, with_weight), **arrays)


def read_cache(fold, with_weight):
    with np.load(cache_path(fold, with_weight)) as handle:
        return {label: (handle[f"s{label}"], handle[f"w{label}"],
                        int(handle[f"n{label}"]))
                for label in range(4)}


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def weighted_auc(pos_score, pos_weight, neg_score, neg_weight):
    """Mann-Whitney AUC with weights; ties contribute 0.5.

    Vectorised over score tie groups: `np.unique(..., return_inverse=True)`
    gives one integer per event, then `bincount` sums the weights per group.
    """
    n_pos, n_neg = pos_score.size, neg_score.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    w_pos = float(np.sum(pos_weight))
    w_neg = float(np.sum(neg_weight))
    if w_pos <= 0.0 or w_neg <= 0.0:
        return float("nan")

    score = np.concatenate([pos_score, neg_score]).astype(np.float64)
    weight = np.concatenate([pos_weight, neg_weight]).astype(np.float64)
    is_pos = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])

    _, group = np.unique(score, return_inverse=True)
    n_groups = int(group.max()) + 1

    pos_group = np.bincount(group, weights=weight * is_pos,
                            minlength=n_groups)
    neg_group = np.bincount(group, weights=weight * (~is_pos),
                            minlength=n_groups)

    # `below` is the positive weight strictly below each (sorted) group.
    below = np.cumsum(pos_group) - pos_group
    # Accumulates P(score_pos < score_neg); the AUC is its complement.
    below_neg = float(np.sum(neg_group * (below + 0.5 * pos_group)))
    return 1.0 - below_neg / (w_pos * w_neg)


def hanley_se(auc, n_pos, n_neg):
    if not np.isfinite(auc) or n_pos <= 0 or n_neg <= 0:
        return float("nan")
    q1 = auc / (2.0 - auc) if auc < 2.0 else 0.0
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc))
    return float(np.sqrt(max(var, 0.0) / (n_pos * n_neg)))


def pair_score(p_i, p_j, raw_p):
    """Common one-vs-one ranking statistic applied to every event.

    Default: r = p_i / (p_i + p_j) = P(class i | class in {i, j}).
    `--raw-p`: r = p_i, i.e. rank by the class-i probability alone.
    """
    if raw_p:
        return p_i
    denom = p_i + p_j
    out = np.full_like(p_i, 0.5, dtype=np.float64)
    np.divide(p_i, denom, out=out, where=denom > 0)
    return out


def auc_pair(fold_data, i, j, raw_p):
    si, wi, _ = fold_data[i]
    sj, wj, _ = fold_data[j]
    # The SAME statistic r is evaluated on both event sets; class i is
    # positive, so true-i events should have larger r than true-j events.
    score_i = pair_score(si[:, i], si[:, j], raw_p)
    score_j = pair_score(sj[:, i], sj[:, j], raw_p)
    return weighted_auc(score_i, wi, score_j, wj)


def auc_ovr(fold_data, i):
    pos_score, pos_w, _ = fold_data[i]
    neg_score = np.concatenate([fold_data[j][0] for j in range(4) if j != i])
    neg_w = np.concatenate([fold_data[j][1] for j in range(4) if j != i])
    return weighted_auc(pos_score[:, i], pos_w,
                        neg_score[:, i], neg_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted", action="store_true",
                        help="Weight AUC by weight_lumi_avg (default: unweighted).")
    parser.add_argument("--raw-p", action="store_true",
                        help="Rank by p_i alone instead of p_i/(p_i+p_j).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-read the ROOT files instead of using the cache.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_data = {}
    for fold in range(4):
        cached = cache_path(fold, args.weighted)
        if not args.no_cache and cached.exists():
            print(f"Loading fold {fold} from cache ({cached.name}) ...")
            fold_data[fold] = read_cache(fold, args.weighted)
        else:
            print(f"Loading fold {fold} (test parity {fold}) ...")
            fold_data[fold] = load_fold_classes(fold, args.weighted)
            save_cache(fold, fold_data[fold], args.weighted)
        counts = ", ".join(
            f"{NAMES[i]}={fold_data[fold][i][2]}" for i in range(4))
        print(f"  test-parity events: {counts}")

    # Pool the four disjoint test parities.
    pooled = {i: (np.concatenate([fold_data[f][i][0] for f in range(4)]),
                  np.concatenate([fold_data[f][i][1] for f in range(4)]),
                  sum(fold_data[f][i][2] for f in range(4)))
              for i in range(4)}

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    per_fold_rows = []
    pooled_rows = []
    for i, j in pairs:
        vals = [auc_pair(fold_data[f], i, j, args.raw_p) for f in range(4)]
        a = auc_pair(pooled, i, j, args.raw_p)
        n_pos = pooled[i][2]
        n_neg = pooled[j][2]
        pooled_rows.append({
            "pair": f"{NAMES[i]} vs {NAMES[j]}",
            "class_i": NAMES[i], "class_j": NAMES[j],
            "auc": a,
            "sigma_auc_hanley": (hanley_se(a, n_pos, n_neg)
                                 if not args.weighted else float("nan")),
            "n_i": n_pos, "n_j": n_neg,
            "fold0": vals[0], "fold1": vals[1],
            "fold2": vals[2], "fold3": vals[3],
            "fold_mean": float(np.mean(vals)),
            "fold_std": float(np.std(vals, ddof=1)),
        })
        for f, v in enumerate(vals):
            per_fold_rows.append({
                "fold": f, "class_i": NAMES[i], "class_j": NAMES[j],
                "auc": v, "n_i": fold_data[f][i][2],
                "n_j": fold_data[f][j][2],
            })

    ovr_rows = []
    for i in range(4):
        a = auc_ovr(pooled, i)
        ovr_rows.append({"class": NAMES[i], "auc_ovr": a,
                         "n_pos": pooled[i][2],
                         "n_neg": sum(pooled[j][2] for j in range(4) if j != i)})

    # Symmetric matrices for the heatmaps / CSV: one per fold + pooled.
    mats = {}
    for f in range(4):
        m = np.full((4, 4), np.nan)
        for i, j in pairs:
            m[i, j] = m[j, i] = auc_pair(fold_data[f], i, j, args.raw_p)
        mats[f] = m
    mat = np.full((4, 4), np.nan)
    for i, j in pairs:
        mat[i, j] = mat[j, i] = auc_pair(pooled, i, j, args.raw_p)

    write_csvs(per_fold_rows, pooled_rows, ovr_rows, mat, args)
    make_figures(mats, mat, args)
    write_report(pooled_rows, ovr_rows, pooled, args)

    print("\nPooled pairwise AUC (out-of-fold test parities)")
    print(f"{'pair':<20}{'AUC':>9}{'sigma':>10}")
    for r in pooled_rows:
        print(f"{r['pair']:<20}{r['auc']:9.5f}"
              f"{r['sigma_auc_hanley']:10.5f}")
    print("\nOne-vs-rest AUC (pooled)")
    for r in ovr_rows:
        print(f"{r['class']:<12}{r['auc_ovr']:9.5f}")
    print(f"\nReport: {OUT_DIR / 'report.md'}")
    print("Figures: pairwise_auc_fold0.png ... pairwise_auc_fold3.png, "
          "pairwise_auc_4folds.png, pairwise_auc_matrix.png")


def write_csvs(per_fold_rows, pooled_rows, ovr_rows, mat, args):
    with (OUT_DIR / "pairwise_auc_per_fold.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_fold_rows[0]))
        writer.writeheader()
        writer.writerows(per_fold_rows)

    with (OUT_DIR / "pairwise_auc_pooled.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled_rows[0]))
        writer.writeheader()
        writer.writerows(pooled_rows)

    with (OUT_DIR / "ovr_auc_pooled.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ovr_rows[0]))
        writer.writeheader()
        writer.writerows(ovr_rows)

    with (OUT_DIR / "auc_matrix.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class"] + NAMES)
        for i, name in enumerate(NAMES):
            writer.writerow([name] + [
                "" if not np.isfinite(mat[i, j]) else f"{mat[i, j]:.6f}"
                for j in range(4)])


def write_report(pooled_rows, ovr_rows, pooled, args):
    rows = [[
        r["class_i"], r["class_j"], f"{r['auc']:.5f}",
        "" if not np.isfinite(r["sigma_auc_hanley"])
        else f"{r['sigma_auc_hanley']:.5f}",
        f"{r['fold_mean']:.5f}", f"{r['fold_std']:.5f}",
        r["n_i"], r["n_j"],
    ] for r in pooled_rows]

    table = ["| class i | class j | pooled AUC | σ (Hanley) | fold mean | fold std "
             "| N_i | N_j |",
             "|---|---|---|---|---|---|---|---|"]
    table += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    table = "\n".join(table)

    ovr_table = ["| class | one-vs-rest AUC | N_pos | N_neg |",
                 "|---|---|---|---|"]
    ovr_table += ["| " + " | ".join(str(x) for x in
                                    [r["class"], f"{r['auc_ovr']:.5f}",
                                     r["n_pos"], r["n_neg"]]) + " |"
                  for r in ovr_rows]
    ovr_table = "\n".join(ovr_table)

    counts = ", ".join(f"{NAMES[i]} = {pooled[i][2]}" for i in range(4))
    minv = min(min(r["auc"], r["fold0"], r["fold1"], r["fold2"], r["fold3"])
               for r in pooled_rows)
    score_note = ("`p_i` (raw)" if args.raw_p
                  else "`p_i / (p_i + p_j)` (conditional pair probability)")
    weight_note = ("weighted by `weight_lumi_avg`" if args.weighted
                   else "unweighted")

    report = f"""# Pairwise AUC for the 4-class GNN

Computed by `tools/pairwise_auc_4class.py` from the out-of-fold scored
trees. Class i is the positive class; the ranking score is {score_note}.
AUC is {weight_note}.

Class mapping (same as the training config `label`):

| label | name | processes |
|---|---|---|
| 0 | bkg | `Bkg_MC/TT`, `Bkg_MC/DYGto2LG_*`, `Bkg_MC/DYJetsToLL_Stitch` |
| 1 | ttH | `Signal/ttH` |
| 2 | VH | `Signal/VH` |
| 3 | ggF_VBF | `Signal/ggF` + `Signal/VBF` |

Only the test parity of each fold is used (`photon_eta_fold % 4 == fold`),
so every event is scored by the model that did not train on it. Pooled
events: {counts}.

## Pairwise (one-vs-one) AUC

{table}

`fold mean ± std` is the spread over the four independent test parities and
is a simple stability check, not an uncertainty on the pooled value.

## One-vs-rest AUC (pooled)

{ovr_table}

The one-vs-rest value uses `p_i` against all other classes. This is the
closest analogue of the framework's `Test_AUC`, but the framework computes it
on its test split **with `sample_weight = weight_lumi_avg`** and takes the
macro average over the four classes; here it is unweighted and pooled over the
four test parities. `tools/training_auc_check.py` reproduces the framework
number to within 5e-4 (`report_training_auc.md`), and is what should be
compared against `training.log`.

## Per-fold heatmaps

`pairwise_auc_fold0.png` … `pairwise_auc_fold3.png` show the same matrix for
each fold separately, and `pairwise_auc_4folds.png` puts the four side by
side. All panels (including the pooled one) share one colour scale, clipped
to the range of the data — the minimum off-diagonal value is about {minv:.3f},
so a 0.5–1.0 scale would render every cell the same colour.

## Reading the matrix

- AUC is symmetric only in the sense that the pair (i, j) and (j, i) use the
  same two event sets; the reported matrix sets them equal, with i < j.
- The weakest pair is `bkg vs ggF_VBF` (0.874): the ggF/VBF class is the
  most background-like, which is consistent with its low `S/B` and small
  `Z²` share in the significance breakdown. It is still well separated at
  the classifier level — this is a relative statement, not a failure.
- The strongest pair is `ttH vs ggF_VBF` (0.992); `ttH vs VH` (0.982) is
  next. Those are the two signal-vs-signal pairs that the 4-class problem
  has to resolve.
- These AUCs describe the 4-class classifier itself. They are **not**
  comparable with the signal significance Z in
  `results_compare_cutbase_vs_gnn/report.md`; high pairwise AUC does not by
  itself imply a high Asimov Z.

## Files

- `pairwise_auc_per_fold.csv`
- `pairwise_auc_pooled.csv`
- `ovr_auc_pooled.csv`
- `auc_matrix.csv`
- `pairwise_auc_fold0.png` … `pairwise_auc_fold3.png` — one heatmap per fold
- `pairwise_auc_4folds.png` — the four folds side by side, shared scale
- `pairwise_auc_matrix.png` — pooled

![pairwise AUC, pooled](pairwise_auc_matrix.png)
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")


def make_figures(mats, pooled_mat, args):
    """One heatmap per fold (shared colour scale) plus the pooled one.

    The colour scale is shared by every panel and clipped to the range of the
    data (0.85 -> 1.0 here): with 0.5 -> 1.0 every cell would be the same dark
    blue and the fold-to-fold differences would be invisible.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
    })

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad(SURFACE)

    off_diag = [pooled_mat[i, j] for i in range(4) for j in range(i + 1, 4)]
    for f in range(4):
        off_diag += [mats[f][i, j] for i in range(4) for j in range(i + 1, 4)]
    off_diag = np.asarray([v for v in off_diag if np.isfinite(v)])
    vmin = max(0.0, np.floor(off_diag.min() * 20.0) / 20.0)
    vmax = 1.0

    # --- four separate figures, one per fold -------------------------------
    for f in range(4):
        fig, ax = plt.subplots(figsize=(6.0, 5.1))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        im = _draw_matrix(ax, mats[f], cmap, vmin, vmax)
        _decorate(ax)
        ax.set_title(
            f"Fold {f} — pairwise AUC (test parity {f} only)",
            loc="left", color=INK, fontsize=11, pad=8,
        )
        fig.text(
            0.012, 0.918,
            "score = p$_i$ / (p$_i$ + p$_j$); class i = row, class j = column",
            fontsize=8.5, color=INK2,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("pairwise AUC", color=INK2, fontsize=9)
        cbar.ax.tick_params(colors=INK2, labelsize=8, length=0)
        cbar.outline.set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT_DIR / f"pairwise_auc_fold{f}.png", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # --- 2x2 small multiples sharing one colour bar -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    fig.patch.set_facecolor(SURFACE)
    for f, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
        im = _draw_matrix(ax, mats[f], cmap, vmin, vmax)
        _decorate(ax)
        ax.set_title(f"Fold {f} (test parity {f})", loc="left",
                     color=INK, fontsize=10, pad=6)
    fig.suptitle("4-class GNN pairwise AUC per fold (out-of-fold)",
                 x=0.012, ha="left", color=INK, fontsize=12)
    fig.text(0.012, 0.935,
             "score = p$_i$ / (p$_i$ + p$_j$); class i = row, class j = column; "
             "shared colour scale",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    cbar_ax = fig.add_axes([0.945, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("pairwise AUC", color=INK2, fontsize=9)
    cbar.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    fig.savefig(OUT_DIR / "pairwise_auc_4folds.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # --- pooled figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 5.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = _draw_matrix(ax, pooled_mat, cmap, vmin, vmax)
    _decorate(ax)
    ax.set_title(
        "4-class GNN pairwise AUC (out-of-fold, pooled)",
        loc="left", color=INK, fontsize=11, pad=8,
    )
    fig.text(
        0.012, 0.918,
        "score = p$_i$ / (p$_i$ + p$_j$); class i = row, class j = column",
        fontsize=8.5, color=INK2,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("pairwise AUC", color=INK2, fontsize=9)
    cbar.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "pairwise_auc_matrix.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _draw_matrix(ax, mat, cmap, vmin, vmax):
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        color=INK2, fontsize=10)
            else:
                value = mat[i, j]
                # White text only where the cell is dark enough for 4.5:1.
                color = "white" if value >= 0.95 else INK
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        color=color, fontsize=9)
    return im


def _decorate(ax):
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(NAMES)
    ax.set_yticklabels(NAMES)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    main()
