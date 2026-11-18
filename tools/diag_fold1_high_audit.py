#!/usr/bin/env python3
"""Plot 4-fold GNN score distributions for validation/test and original categories.

For each fold:
  1. Validation and test score distributions: unweighted counts and weight_lumi yield.
  2. The same distributions restricted to the fold's high category (score > c2).

For the five original analysis categories:
  Aggregate the four independent test parities (each event is counted once, using
  the score from its corresponding fold model), then plot signal/background score
  distributions, both unweighted and weighted. These are GNN-score plots grouped
  by the five original categories; they are not a cut-based classifier score.
"""

from pathlib import Path
import warnings

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
OUT_DIR = BASE_DIR / "tools/results_4fold_pt_only/score_distributions"

FOLD_DIRS = [
    BASE_DIR / f"root_p2root_v2_pt_only_scored_ttH_pt_only_4fold_fold{i}"
    for i in range(4)
]

# The model in fold k is evaluated on test parity k and validation parity below.
TEST_PARITY = {0: 0, 1: 1, 2: 2, 3: 3}
VALID_PARITY = {0: 1, 1: 2, 2: 3, 3: 0}

# Bayesian-selected boundaries from the current 2026-10-27 results.
CHOSEN = {
    0: (0.938372, 0.986765),
    1: (0.949845, 0.992027),
    2: (0.912493, 0.993509),
    3: (0.949478, 0.996749),
}

TREE = "merged_SR_Sideband"
WEIGHT = "weight_lumi"
MASS = "m_llg"
SCORE = "ttH_gnn_score"
IS_SR = "is_SR"
TREE_TAG = "source_tree_tag"
FOLDVAR = "photon_eta_fold"
REQUIRED_BRANCHES = [WEIGHT, MASS, SCORE, IS_SR, TREE_TAG, FOLDVAR]

MASS_LO, MASS_HI = 120.0, 130.0
SCORE_BINS = np.linspace(0.0, 1.0, 51)
N_HIGH_BINS = 30

SR_TAG_MAP = {
    0: "ttHhad",
    2: "ttHlep",
    4: "untag",
    6: "VH3l",
    8: "VHpTmiss",
}
VALID_SR_TAGS = np.array(list(SR_TAG_MAP.keys()))
RUN3 = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]
SIGNAL_GROUP = ("Signal/ttH", RUN3)
BACKGROUND_GROUPS = [
    ("Signal/VH", RUN3),
    ("Signal/ggF", RUN3),
    ("Signal/VBF", RUN3),
    ("Bkg_MC/TT", RUN3),
    ("Bkg_MC/DYGto2LG_10to100", RUN3),
]

warnings.filterwarnings("ignore", category=RuntimeWarning)


def empty_arrays():
    return {branch: np.array([], dtype=float) for branch in REQUIRED_BRANCHES}


def load_file(path: Path):
    """Read required branches from one ROOT file; fail loudly on malformed files."""
    with uproot.open(path) as root_file:
        if TREE not in root_file:
            raise RuntimeError(f"Tree '{TREE}' is missing in {path}")
        tree = root_file[TREE]
        available = set(tree.keys())
        missing = [branch for branch in REQUIRED_BRANCHES if branch not in available]
        if missing:
            raise RuntimeError(f"Missing required branches in {path}: {missing}")
        arrays = tree.arrays(REQUIRED_BRANCHES, library="np", how=dict)
        return {branch: np.asarray(arrays[branch]) for branch in REQUIRED_BRANCHES}


def concatenate_arrays(chunks):
    chunks = [chunk for chunk in chunks if chunk and chunk[WEIGHT].size > 0]
    if not chunks:
        return empty_arrays()
    return {
        branch: np.concatenate([chunk[branch] for chunk in chunks])
        for branch in REQUIRED_BRANCHES
    }


def load_group(fold_dir: Path, subdir: str, years):
    chunks = []
    for year in years:
        path = fold_dir / subdir / f"{year}.root"
        if not path.exists():
            # Some samples may not be present for every year. Keep the omission visible.
            print(f"  [skip missing file] {path}")
            continue
        chunks.append(load_file(path))
    return concatenate_arrays(chunks)


def load_fold(fold_dir: Path):
    signal = load_group(fold_dir, SIGNAL_GROUP[0], SIGNAL_GROUP[1])
    background_chunks = [load_group(fold_dir, subdir, years)
                         for subdir, years in BACKGROUND_GROUPS]
    background = concatenate_arrays(background_chunks)
    return signal, background


def select_sr_parity(arr, parity):
    """Use the same SR, mass, category, finite-value and parity selection as significance."""
    n = arr[WEIGHT].size
    if n == 0:
        return empty_arrays()
    mask = (
        (arr[IS_SR] == 1)
        & (arr[MASS] > MASS_LO)
        & (arr[MASS] < MASS_HI)
        & np.isin(arr[TREE_TAG], VALID_SR_TAGS)
        & np.isfinite(arr[WEIGHT])
        & np.isfinite(arr[SCORE])
        & np.isfinite(arr[FOLDVAR])
        & ((arr[FOLDVAR] % 4) == parity)
    )
    return {branch: arr[branch][mask] for branch in REQUIRED_BRANCHES}


def draw_two_histograms(ax, sig, bkg, bins, weighted=False, title=""):
    """Draw signal/background score distributions; weighted mode uses weight_lumi."""
    sig_score = sig[SCORE]
    bkg_score = bkg[SCORE]
    if weighted:
        sig_w = sig[WEIGHT]
        bkg_w = bkg[WEIGHT]
        ax.hist(sig_score, bins=bins, weights=sig_w, histtype="step", linewidth=1.6,
                label=f"Signal (sum w={np.sum(sig_w):.4g})")
        ax.hist(bkg_score, bins=bins, weights=bkg_w, histtype="step", linewidth=1.6,
                label=f"Background (sum w={np.sum(bkg_w):.4g})")
        ax.set_ylabel("Weighted yield per bin (weight_lumi)")
    else:
        ax.hist(sig_score, bins=bins, histtype="step", linewidth=1.6,
                label=f"Signal (N={sig_score.size})")
        ax.hist(bkg_score, bins=bins, histtype="step", linewidth=1.6,
                label=f"Background (N={bkg_score.size})")
        ax.set_ylabel("Unweighted event count")
    ax.set_xlabel("ttH_gnn_score")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_fold_full(fold, sig_val, bkg_val, sig_test, bkg_test, c1, c2):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = [
        (axes[0, 0], sig_val, bkg_val, False, "Validation — unweighted"),
        (axes[0, 1], sig_val, bkg_val, True, "Validation — weight_lumi"),
        (axes[1, 0], sig_test, bkg_test, False, "Test — unweighted"),
        (axes[1, 1], sig_test, bkg_test, True, "Test — weight_lumi"),
    ]
    for ax, sig, bkg, weighted, title in panels:
        draw_two_histograms(ax, sig, bkg, SCORE_BINS, weighted, title)
        ax.axvline(c1, color="black", linestyle="--", linewidth=1.2, label=f"c1={c1:.4f}")
        ax.axvline(c2, color="black", linestyle="-.", linewidth=1.2, label=f"c2={c2:.4f}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fold {fold}: full GNN score distributions (SR, 120 < m_llg < 130 GeV)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / f"fold{fold}_validation_test_score.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def plot_fold_high(fold, sig_val, bkg_val, sig_test, bkg_test, c2):
    """Plot score shapes after applying this fold's high-category requirement score > c2."""
    high_bins = np.linspace(c2, 1.0, N_HIGH_BINS + 1)

    def high_only(arr):
        mask = arr[SCORE] > c2
        return {branch: arr[branch][mask] for branch in REQUIRED_BRANCHES}

    sig_val_h, bkg_val_h = high_only(sig_val), high_only(bkg_val)
    sig_test_h, bkg_test_h = high_only(sig_test), high_only(bkg_test)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = [
        (axes[0, 0], sig_val_h, bkg_val_h, False, "Validation HIGH — unweighted"),
        (axes[0, 1], sig_val_h, bkg_val_h, True, "Validation HIGH — weight_lumi"),
        (axes[1, 0], sig_test_h, bkg_test_h, False, "Test HIGH — unweighted"),
        (axes[1, 1], sig_test_h, bkg_test_h, True, "Test HIGH — weight_lumi"),
    ]
    for ax, sig, bkg, weighted, title in panels:
        draw_two_histograms(ax, sig, bkg, high_bins, weighted, title)
        ax.axvline(c2, color="black", linestyle="-.", linewidth=1.2, label=f"c2={c2:.4f}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fold {fold}: high-category score distributions (score > {c2:.4f})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / f"fold{fold}_validation_test_HIGH_score.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def add_category_histograms(accumulator, arr, is_signal, bins):
    """Accumulate histograms by original analysis tag."""
    score = arr[SCORE]
    weight = arr[WEIGHT]
    tags = arr[TREE_TAG]
    for tag, name in SR_TAG_MAP.items():
        mask = tags == tag
        counts = np.histogram(score[mask], bins=bins)[0]
        yields = np.histogram(score[mask], bins=bins, weights=weight[mask])[0]
        prefix = "sig" if is_signal else "bkg"
        accumulator[name][f"{prefix}_count"] += counts
        accumulator[name][f"{prefix}_weight"] += yields


def plot_original_categories(accumulator, weighted=False):
    """Plot five original categories after aggregating all four test parities."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    axes = axes.ravel()
    edges = SCORE_BINS
    for ax, (tag, name) in zip(axes, SR_TAG_MAP.items()):
        data = accumulator[name]
        if weighted:
            sig_y = data["sig_weight"]
            bkg_y = data["bkg_weight"]
            ylabel = "Weighted yield per bin (weight_lumi)"
            sig_label = f"Signal (sum w={sig_y.sum():.4g})"
            bkg_label = f"Background (sum w={bkg_y.sum():.4g})"
        else:
            sig_y = data["sig_count"]
            bkg_y = data["bkg_count"]
            ylabel = "Unweighted event count"
            sig_label = f"Signal (N={int(sig_y.sum())})"
            bkg_label = f"Background (N={int(bkg_y.sum())})"
        ax.stairs(sig_y, edges, linewidth=1.6, label=sig_label)
        ax.stairs(bkg_y, edges, linewidth=1.6, label=bkg_label)
        ax.set_title(f"{name} (tag={tag})")
        ax.set_xlabel("ttH_gnn_score")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    kind = "weight_lumi" if weighted else "unweighted"
    fig.suptitle(
        f"Original five analysis categories — {kind}\n"
        "Four test parities aggregated; each parity scored by its corresponding fold model"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / f"five_categories_score_{kind}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print("Required selection: is_SR==1, 120<m_llg<130, valid original category, finite score/weight")
    print("For the five-category overview, only each fold's TEST parity is used to avoid double-counting.")

    # Accumulators for the five original categories, combining each event exactly once
    # via the test parity assigned to its corresponding fold model.
    category_acc = {
        name: {
            "sig_count": np.zeros(len(SCORE_BINS) - 1),
            "sig_weight": np.zeros(len(SCORE_BINS) - 1),
            "bkg_count": np.zeros(len(SCORE_BINS) - 1),
            "bkg_weight": np.zeros(len(SCORE_BINS) - 1),
        }
        for name in SR_TAG_MAP.values()
    }

    for fold in range(4):
        fold_dir = FOLD_DIRS[fold]
        print(f"\n[Fold {fold}] Loading {fold_dir}")
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory does not exist: {fold_dir}")
        signal, background = load_fold(fold_dir)

        val_parity = VALID_PARITY[fold]
        test_parity = TEST_PARITY[fold]
        sig_val = select_sr_parity(signal, val_parity)
        bkg_val = select_sr_parity(background, val_parity)
        sig_test = select_sr_parity(signal, test_parity)
        bkg_test = select_sr_parity(background, test_parity)
        c1, c2 = CHOSEN[fold]

        print(
            f"  val parity={val_parity}: Nsig={sig_val[SCORE].size}, Nbkg={bkg_val[SCORE].size}; "
            f"test parity={test_parity}: Nsig={sig_test[SCORE].size}, Nbkg={bkg_test[SCORE].size}; "
            f"c1={c1:.6f}, c2={c2:.6f}"
        )

        plot_fold_full(fold, sig_val, bkg_val, sig_test, bkg_test, c1, c2)
        plot_fold_high(fold, sig_val, bkg_val, sig_test, bkg_test, c2)

        # Aggregate only the independent test parity for the original-category plots.
        add_category_histograms(category_acc, sig_test, True, SCORE_BINS)
        add_category_histograms(category_acc, bkg_test, False, SCORE_BINS)

    plot_original_categories(category_acc, weighted=False)
    plot_original_categories(category_acc, weighted=True)
    print("\nDone. Created 10 plots in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
