#!/usr/bin/env python3
"""
Plot training curves (loss/AUC/accuracy) for fold0 and fold1,
plus OOF GNN score distributions for signal and background.

Output: tools/results/training_and_scores.png
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# =============================================================================
# Config
# =============================================================================

LOG0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "trainings/hzg/ttH_vs_other_run3_0828_fold0/training.log"
)
LOG1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "trainings/hzg/ttH_vs_other_run3_0828_fold1/training.log"
)

DATA_DIR_FOLD0 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_scored_ttH_vs_other_run3v2_fold0_0828"
)
DATA_DIR_FOLD1 = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2_scored_ttH_vs_other_run3v2_fold1_0828"
)

OUT_DIR = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results")

TREE = "merged_SR_Sideband"
WEIGHT = "weight_lumi"
SCORE  = "ttH_gnn_score"
MASS   = "m_llg"
IS_SR  = "is_SR"
TAG    = "source_tree_tag"
PHI    = "photon_eta_fold"

VALID_TAGS = np.array([0, 2, 4, 6, 8])
MASS_LO, MASS_HI = 120.0, 130.0
C1 = 0.928356
C2 = 0.994757
RUN3 = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]

SIGNAL_GROUP = ("Signal/ttH", RUN3)
OTHER_GROUPS = [
    ("Signal/VH",                RUN3),
    ("Signal/ggF",               RUN3),
    ("Signal/VBF",               RUN3),
    ("Bkg_MC/TT",                RUN3),
    ("Bkg_MC/DYGto2LG_10to100", RUN3),
]

BRANCHES = [WEIGHT, SCORE, MASS, IS_SR, TAG, PHI]

# =============================================================================
# Parse training log
# =============================================================================

def parse_log(path):
    ep, loss, acc, tl, tauc, t = [], [], [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+).*?Loss\s+([\d.]+).*?Accuracy\s+([\d.]+)"
        r".*?Test_Loss\s+([\d.]+).*?Test_AUC\s+([\d.]+).*?Time\s+([\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
                tl.append(float(m.group(4)))
                tauc.append(float(m.group(5)))
                t.append(float(m.group(6)))

    cum_t = np.cumsum(t) / 3600.0   # hours
    return {
        "epoch":    np.array(ep),
        "loss":     np.array(loss),
        "acc":      np.array(acc),
        "test_loss": np.array(tl),
        "test_auc": np.array(tauc),
        "cum_time": cum_t,
    }


# =============================================================================
# Load OOF scores
# =============================================================================

def load_file(path):
    try:
        with uproot.open(str(path)) as f:
            if TREE not in f:
                return None
            tree = f[TREE]
            if any(b not in tree.keys() for b in BRANCHES):
                return None
            return tree.arrays(BRANCHES, library="np")
    except Exception as e:
        print(f"WARNING: {path}: {e}", file=sys.stderr)
        return None


def load_group(data_dir, subdir, years):
    pieces = []
    for yr in years:
        path = data_dir / subdir / f"{yr}.root"
        if not path.exists():
            continue
        arr = load_file(path)
        if arr is not None and arr[WEIGHT].size > 0:
            pieces.append(arr)
    if not pieces:
        return None
    out = {}
    for br in BRANCHES:
        out[br] = np.concatenate([p[br] for p in pieces])
    return out


def oof_mask(arr, parity):
    return (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TAG], VALID_TAGS)
        & (arr[PHI] % 2 == parity)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
    )


def load_oof_scores():
    print("Loading OOF scores for score distribution...")

    # fold0 test (parity=0) from fold0 scored files
    sig0 = load_group(DATA_DIR_FOLD0, *SIGNAL_GROUP)
    bkg0_parts = [load_group(DATA_DIR_FOLD0, s, e) for s, e in OTHER_GROUPS]
    bkg0_parts = [x for x in bkg0_parts if x is not None]

    # fold1 test (parity=1) from fold1 scored files
    sig1 = load_group(DATA_DIR_FOLD1, *SIGNAL_GROUP)
    bkg1_parts = [load_group(DATA_DIR_FOLD1, s, e) for s, e in OTHER_GROUPS]
    bkg1_parts = [x for x in bkg1_parts if x is not None]

    def concat_parts(parts):
        if not parts:
            return {br: np.array([]) for br in BRANCHES}
        out = {}
        for br in BRANCHES:
            out[br] = np.concatenate([p[br] for p in parts])
        return out

    bkg0 = concat_parts(bkg0_parts)
    bkg1 = concat_parts(bkg1_parts)

    # apply OOF mask
    def masked(arr, parity):
        if arr is None:
            return np.array([]), np.array([])
        m = oof_mask(arr, parity)
        return arr[WEIGHT][m], arr[SCORE][m]

    ws0, ss0 = masked(sig0, 0)
    wb0, sb0 = masked(bkg0, 0)
    ws1, ss1 = masked(sig1, 1)
    wb1, sb1 = masked(bkg1, 1)

    sig_w = np.concatenate([ws0, ws1])
    sig_s = np.concatenate([ss0, ss1])
    bkg_w = np.concatenate([wb0, wb1])
    bkg_s = np.concatenate([sb0, sb1])

    return sig_w, sig_s, bkg_w, bkg_s


# =============================================================================
# Plot
# =============================================================================

# =============================================================================
# Plot helpers
# =============================================================================

def plot_fold_training(log, fold_name, out_path):
    """1x3: Loss | Test AUC | Train Accuracy for one fold."""
    color = "#1f77b4" if fold_name == "fold0" else "#ff7f0e"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"ttH vs Other — {fold_name} training curves", fontsize=12)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(log["epoch"], log["loss"],      color=color, label="train loss")
    ax.plot(log["epoch"], log["test_loss"], color=color, ls="--", label="test loss")
    best_ep = log["epoch"][np.argmin(log["test_loss"])]
    best_tl = np.min(log["test_loss"])
    ax.axvline(best_ep, color=color, ls=":", alpha=0.7)
    ax.annotate(
        f"ep{best_ep}\n{best_tl:.4f}",
        xy=(best_ep, best_tl), xytext=(best_ep + 1, best_tl * 1.05),
        fontsize=8, color=color,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss  (solid=train, dashed=test)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Test AUC ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(log["epoch"], log["test_auc"], color=color, label="test AUC")
    best_ep_auc = log["epoch"][np.argmax(log["test_auc"])]
    best_auc    = np.max(log["test_auc"])
    ax.axvline(best_ep_auc, color=color, ls=":", alpha=0.7)
    ax.annotate(
        f"ep{best_ep_auc}\n{best_auc:.4f}",
        xy=(best_ep_auc, best_auc), xytext=(best_ep_auc + 1, best_auc - 0.003),
        fontsize=8, color=color,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test AUC")
    ax.set_title("Test AUC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Train Accuracy ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(log["epoch"], log["acc"], color=color, label="train acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")
    ax.set_title("Train Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

def plot_oof_scores(sig_w, sig_s, bkg_w, bkg_s, out_path):
    """
    Plot OOF GNN score distributions without event weights.
    Single panel: unweighted, normalized by event count.
    Show c1 and c2 as vertical lines.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 1, 51)

    # unweighted normalization by event count
    sig_norm = np.ones(sig_s.size) / sig_s.size if sig_s.size > 0 else None
    bkg_norm = np.ones(bkg_s.size) / bkg_s.size if bkg_s.size > 0 else None

    # fresh/light colors
    sig_color = "#7fcdbb"   # soft teal
    bkg_color = "#8da0cb"   # soft blue-violet

    ax.hist(
        sig_s,
        bins=bins,
        # weights=sig_norm,
        histtype="stepfilled",
        alpha=0.5,
        color=sig_color,
        edgecolor=sig_color,
        label=f"Signal (N={sig_s.size})",
    )

    ax.hist(
        bkg_s,
        bins=bins,
        # weights=bkg_norm,
        histtype="step",
        linewidth=2.0,
        color=bkg_color,
        label=f"Bkg (N={bkg_s.size})",
    )

    # draw cuts
    ax.axvline(
        C1,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"c1 = {C1:.6f}",
    )
    ax.axvline(
        C2,
        color="#222222",
        linestyle=":",
        linewidth=1.8,
        label=f"c2 = {C2:.6f}",
    )

    ax.set_xlabel("GNN score")
    ax.set_ylabel("Fraction / bin")
    ax.set_title("signal vs bkg score")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)



def make_figure(log0, log1, sig_w, sig_s, bkg_w, bkg_s):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fold_training(log0, "fold0", OUT_DIR / "fold0_training.png")
    plot_fold_training(log1, "fold1", OUT_DIR / "fold1_training.png")
    plot_oof_scores(sig_w, sig_s, bkg_w, bkg_s, OUT_DIR / "oof_scores.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Parsing training logs...")
    log0 = parse_log(LOG0)
    log1 = parse_log(LOG1)
    print(f"  fold0: {len(log0['epoch'])} epochs, best test_loss={min(log0['test_loss']):.4f} (ep{log0['epoch'][np.argmin(log0['test_loss'])]}), best AUC={max(log0['test_auc']):.4f}")
    print(f"  fold1: {len(log1['epoch'])} epochs, best test_loss={min(log1['test_loss']):.4f} (ep{log1['epoch'][np.argmin(log1['test_loss'])]}), best AUC={max(log1['test_auc']):.4f}")

    sig_w, sig_s, bkg_w, bkg_s = load_oof_scores()
    print(f"OOF  signal: N={sig_s.size}, Y={np.sum(sig_w):.4f}")
    print(f"OOF  bkg   : N={bkg_s.size}, Y={np.sum(bkg_w):.2f}")

    make_figure(log0, log1, sig_w, sig_s, bkg_w, bkg_s)


if __name__ == "__main__":
    main()
