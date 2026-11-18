#!/usr/bin/env python3
"""
Inclusive CR alpha + ttH signal yield diagnostic.

No low/high SB split.
No analysis category split.
No score-region split.
No significance calculation.

Background:
    alpha = N_Data_SB / N_CR_SB
    B = alpha * N_CR_SR

Signal:
    S = sum(weight_lumi)
    sigma_S = sqrt(sum(weight_lumi**2))

Each model fold k uses only TEST parity:
    photon_eta_fold % 4 == k
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")

# Nominal Data
DATA_BASE = (
    REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"
)

# CR Data
CR_BASE = (
    REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold"
)

# Signal MC
MC_BASE = (
    REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
)

TREE_NOM = "merged_SR_Sideband"
TREE_CR = "merged_CR_lowIDMVA"

MASS = "m_llg"
WEIGHT = "weight_lumi"
FOLDVAR = "photon_eta_fold"

ERAS = [
    "2016preVFP",
    "2016postVFP",
    "2017",
    "2018",
    "2022preEE",
    "2022postEE",
    "2023preBPix",
    "2023postBPix",
    "2024",
    "2025",
    "2026",
]

# Mass windows define SR/SB everywhere (no tag / is_SR restriction).
MASS_SR_LO = 120.0
MASS_SR_HI = 130.0

MASS_SB_LOW_LO = 110.0
MASS_SB_HIGH_HI = 180.0


def in_sr(mass):
    return (mass > MASS_SR_LO) & (mass < MASS_SR_HI)


def in_sb(mass):
    low = (mass >= MASS_SB_LOW_LO) & (mass < MASS_SR_LO)
    high = (mass >= MASS_SR_HI) & (mass <= MASS_SB_HIGH_HI)
    return low | high


# =============================================================================
# 2. ROOT loading
# =============================================================================

def read_tree(path, tree_name, branches):

    with uproot.open(path) as f:

        if tree_name not in f:
            raise KeyError(
                f"{tree_name} not found in {path}"
            )

        tree = f[tree_name]

        missing = [
            b for b in branches
            if b not in tree
        ]

        if missing:
            raise KeyError(
                f"Missing branches {missing} in {path}"
            )

        return tree.arrays(
            branches,
            library="np"
        )


def load_data_population(model_fold, sample, missing_files):

    if sample == "nominal":

        base = (
            DATA_BASE.parent /
            f"{DATA_BASE.name}{model_fold}"
        )

        tree_name = TREE_NOM

    elif sample == "cr":

        base = (
            CR_BASE.parent /
            f"{CR_BASE.name}{model_fold}"
        )

        tree_name = TREE_CR

    else:
        raise ValueError(sample)

    mass_parts = []
    parity_parts = []

    for era in ERAS:

        path = (
            base / "Data" / "Data" /
            f"{era}.root"
        )

        if not path.exists():
            missing_files.append(str(path))
            continue

        if sample == "nominal":

            arr = read_tree(
                path,
                tree_name,
                [MASS, FOLDVAR]
            )

            mass = np.asarray(
                arr[MASS], dtype=np.float64
            )

            parity = (
                np.asarray(arr[FOLDVAR], dtype=np.int32) % 4
            )

            # Nominal Data SB: pure mass window (no is_SR, no tag restriction).
            # is_SR marks which analysis tree an event came from, not the mass
            # sideband; the original v3 trees have wider boundaries than the
            # analysis window, so it does not match the mass window.
            keep = (
                np.isfinite(mass)
                & in_sb(mass)
            )

        else:

            arr = read_tree(
                path,
                tree_name,
                [MASS, FOLDVAR]
            )

            mass = np.asarray(
                arr[MASS], dtype=np.float64
            )

            parity = (
                np.asarray(arr[FOLDVAR], dtype=np.int32) % 4
            )

            # CR: pure mass window (no tag restriction). SR vs SB membership is
            # decided downstream by in_sr/in_sb, not by source_tree_tag.
            keep = (
                np.isfinite(mass)
                & (in_sr(mass) | in_sb(mass))
            )

        if np.any(keep):

            mass_parts.append(mass[keep])
            parity_parts.append(parity[keep])

    if not mass_parts:

        return {
            "mass": np.array([], dtype=np.float64),
            "parity": np.array([], dtype=np.int32),
        }

    return {
        "mass": np.concatenate(mass_parts),
        "parity": np.concatenate(parity_parts),
    }


# =============================================================================
# 3. Load ttH signal MC
# =============================================================================

def load_tth_signal(model_fold, missing_files):

    base = (
        MC_BASE.parent /
        f"{MC_BASE.name}{model_fold}"
    )

    mass_parts = []
    weight_parts = []
    parity_parts = []

    for era in ERAS:

        path = (
            base / "Signal" / "ttH" /
            f"{era}.root"
        )

        if not path.exists():
            missing_files.append(str(path))
            continue

        arr = read_tree(
            path,
            TREE_NOM,
            [
                MASS,
                WEIGHT,
                FOLDVAR,
            ]
        )

        mass = np.asarray(
            arr[MASS], dtype=np.float64
        )

        weight = np.asarray(
            arr[WEIGHT], dtype=np.float64
        )

        parity = (
            np.asarray(arr[FOLDVAR], dtype=np.int32) % 4
        )

        # ttH signal selection: pure Higgs mass window (no SR-tree tag
        # restriction). source_tree_tag marks the analysis tree, not the mass
        # window; SB-tree events can have m_llg in the SR window and must be
        # kept as signal.
        keep = (
            np.isfinite(mass)
            & np.isfinite(weight)
            & in_sr(mass)
        )

        if np.any(keep):

            mass_parts.append(mass[keep])
            weight_parts.append(weight[keep])
            parity_parts.append(parity[keep])

    if not mass_parts:

        return {
            "mass": np.array([], dtype=np.float64),
            "weight": np.array([], dtype=np.float64),
            "parity": np.array([], dtype=np.int32),
        }

    return {
        "mass": np.concatenate(mass_parts),
        "weight": np.concatenate(weight_parts),
        "parity": np.concatenate(parity_parts),
    }


def select_parity(population, parity):

    mask = population["parity"] == parity

    return {
        key: values[mask]
        for key, values in population.items()
    }


# =============================================================================
# 4. Signal yield
# =============================================================================

def signal_yield_info(weights):

    weights = np.asarray(
        weights, dtype=np.float64
    )

    if weights.size == 0:

        return {
            "N_S_events": 0,
            "S_yield": 0.0,
            "S_sumw2": 0.0,
            "S_stat_err": 0.0,
            "S_Neff": 0.0,
            "S_Nnegative": 0,
        }

    sumw = float(np.sum(weights))
    sumw2 = float(np.sum(weights ** 2))

    neff = (
        sumw ** 2 / sumw2
        if sumw2 > 0
        else 0.0
    )

    return {
        "N_S_events": int(weights.size),
        "S_yield": sumw,
        "S_sumw2": sumw2,
        "S_stat_err": float(np.sqrt(sumw2)),
        "S_Neff": float(neff),
        "S_Nnegative": int(
            np.count_nonzero(weights < 0)
        ),
    }


# =============================================================================
# 5. Background transfer factor
# =============================================================================

def safe_ratio(numerator, denominator):

    if denominator <= 0:
        return np.nan

    return numerator / denominator


def evaluate_fold(
    fold,
    nominal_data,
    cr_data,
    signal
):

    # Nominal Data SB
    n_data_sb = int(
        np.count_nonzero(
            in_sb(nominal_data["mass"])
        )
    )

    # CR SB
    n_cr_sb = int(
        np.count_nonzero(
            in_sb(cr_data["mass"])
        )
    )

    # CR SR
    n_cr_sr = int(
        np.count_nonzero(
            in_sr(cr_data["mass"])
        )
    )

    # Transfer factor
    alpha = safe_ratio(
        n_data_sb,
        n_cr_sb
    )

    # Background prediction
    b_est = (
        alpha * n_cr_sr
        if np.isfinite(alpha)
        else np.nan
    )

    # Approximate statistical uncertainty
    if (
        np.isfinite(b_est)
        and b_est > 0
        and n_data_sb > 0
        and n_cr_sb > 0
        and n_cr_sr > 0
    ):

        sigma_b = b_est * np.sqrt(
            1.0 / n_data_sb
            + 1.0 / n_cr_sb
            + 1.0 / n_cr_sr
        )

    else:
        sigma_b = np.nan

    row = {

        "fold": fold,
        "test_parity": fold,

        "N_Data_SB": n_data_sb,
        "N_CR_SB": n_cr_sb,
        "N_CR_SR": n_cr_sr,

        "alpha": alpha,

        "B_estimate": b_est,
        "B_stat_err": sigma_b,
    }

    # Add signal yield information
    row.update(
        signal_yield_info(signal["weight"])
    )

    return row


# =============================================================================
# 6. Combine four folds
# =============================================================================

def make_total_row(rows):

    # Combine raw counts first
    n_data_sb = sum(
        r["N_Data_SB"] for r in rows
    )

    n_cr_sb = sum(
        r["N_CR_SB"] for r in rows
    )

    n_cr_sr = sum(
        r["N_CR_SR"] for r in rows
    )

    # Pooled alpha
    alpha = safe_ratio(
        n_data_sb,
        n_cr_sb
    )

    # Inclusive background prediction
    b_est = (
        alpha * n_cr_sr
        if np.isfinite(alpha)
        else np.nan
    )

    if (
        np.isfinite(b_est)
        and b_est > 0
        and n_data_sb > 0
        and n_cr_sb > 0
        and n_cr_sr > 0
    ):

        sigma_b = b_est * np.sqrt(
            1.0 / n_data_sb
            + 1.0 / n_cr_sb
            + 1.0 / n_cr_sr
        )

    else:
        sigma_b = np.nan

    # Combine signal yields
    sumw = sum(
        r["S_yield"] for r in rows
    )

    sumw2 = sum(
        r["S_sumw2"] for r in rows
    )

    n_signal = sum(
        r["N_S_events"] for r in rows
    )

    neff = (
        sumw ** 2 / sumw2
        if sumw2 > 0
        else 0.0
    )

    nnegative = sum(
        r["S_Nnegative"] for r in rows
    )

    return {

        "fold": "TOTAL",
        "test_parity": "0+1+2+3",

        "N_Data_SB": n_data_sb,
        "N_CR_SB": n_cr_sb,
        "N_CR_SR": n_cr_sr,

        "alpha": alpha,

        "B_estimate": b_est,
        "B_stat_err": sigma_b,

        "N_S_events": n_signal,

        "S_yield": sumw,
        "S_sumw2": sumw2,
        "S_stat_err": float(np.sqrt(sumw2)),

        "S_Neff": neff,
        "S_Nnegative": nnegative,
    }


# =============================================================================
# 7. Print results
# =============================================================================

def fmt(value, digits=5):

    if isinstance(value, (float, np.floating)):

        if np.isfinite(value):
            return f"{value:.{digits}f}"

        return "n/a"

    return str(value)


def print_results(rows, total):

    print("\n" + "=" * 145)
    print("INCLUSIVE CR BACKGROUND + ttH SIGNAL YIELD")
    print("=" * 145)

    header = (
        f"{'Fold':>7}"
        f"{'N_Data_SB':>11}"
        f"{'N_CR_SB':>10}"
        f"{'N_CR_SR':>10}"
        f"{'alpha':>10}"
        f"{'B':>12}"
        f"{'errB':>11}"
        f"{'N_sig':>11}"
        f"{'S yield':>13}"
        f"{'errS':>12}"
        f"{'NeffS':>11}"
    )

    print(header)
    print("-" * len(header))

    for row in rows + [total]:

        print(
            f"{str(row['fold']):>7}"
            f"{row['N_Data_SB']:11d}"
            f"{row['N_CR_SB']:10d}"
            f"{row['N_CR_SR']:10d}"
            f"{fmt(row['alpha'], 5):>10}"
            f"{fmt(row['B_estimate'], 4):>12}"
            f"{fmt(row['B_stat_err'], 4):>11}"
            f"{row['N_S_events']:11d}"
            f"{fmt(row['S_yield'], 5):>13}"
            f"{fmt(row['S_stat_err'], 5):>12}"
            f"{fmt(row['S_Neff'], 1):>11}"
        )

    print("-" * len(header))

    print("\nSignal:")
    print("  S = sum(weight_lumi)")
    print("  sigma_S = sqrt(sum(weight_lumi^2))")

    print("\nBackground:")
    print("  alpha = N_Data_SB / N_CR_SB")
    print("  B = alpha * N_CR_SR")


# =============================================================================
# 8. CSV output
# =============================================================================

FIELD_ORDER = [
    "fold",
    "test_parity",
    "N_Data_SB",
    "N_CR_SB",
    "N_CR_SR",
    "alpha",
    "B_estimate",
    "B_stat_err",
    "N_S_events",
    "S_yield",
    "S_stat_err",
    "S_sumw2",
    "S_Neff",
    "S_Nnegative",
]


def write_csv(path, rows):

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELD_ORDER
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                key: row.get(key, "")
                for key in FIELD_ORDER
            })


# =============================================================================
# 9. Plots
# =============================================================================

def plot_event_counts(rows, outdir):

    labels = [
        f"Fold {r['fold']}"
        for r in rows
    ]

    x = np.arange(len(rows))
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    series = [
        ("N_Data_SB", "Nominal Data SB"),
        ("N_CR_SB", "CR Data SB"),
        ("N_CR_SR", "CR Data SR"),
    ]

    for i, (key, label) in enumerate(series):

        offset = (i - 1) * width

        ax.bar(
            x + offset,
            [r[key] for r in rows],
            width=width,
            label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Events")
    ax.set_title("Inclusive Data and CR event counts")

    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        outdir / "event_counts_by_fold.png",
        dpi=170
    )

    plt.close(fig)


def plot_alpha(rows, total, outdir):

    all_rows = rows + [total]

    labels = [
        f"Fold {r['fold']}"
        for r in rows
    ] + ["Combined"]

    values = [
        r["alpha"]
        for r in all_rows
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.bar(
        np.arange(len(values)),
        values
    )

    ax.set_xticks(
        np.arange(len(values))
    )

    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Inclusive CR transfer factor")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        outdir / "alpha_by_fold_and_total.png",
        dpi=170
    )

    plt.close(fig)


def plot_b(rows, total, outdir):

    all_rows = rows + [total]

    labels = [
        f"Fold {r['fold']}"
        for r in rows
    ] + ["Combined"]

    values = [
        r["B_estimate"]
        for r in all_rows
    ]

    errors = [
        r["B_stat_err"]
        for r in all_rows
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.bar(
        np.arange(len(values)),
        values,
        yerr=errors,
        capsize=4
    )

    ax.set_xticks(
        np.arange(len(values))
    )

    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$B=\alpha N_{\mathrm{CR,SR}}$")
    ax.set_title("Inclusive background prediction")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        outdir / "background_prediction_by_fold_and_total.png",
        dpi=170
    )

    plt.close(fig)


def plot_signal_yield(rows, total, outdir):

    all_rows = rows + [total]

    labels = [
        f"Fold {r['fold']}"
        for r in rows
    ] + ["Combined"]

    values = [
        r["S_yield"]
        for r in all_rows
    ]

    errors = [
        r["S_stat_err"]
        for r in all_rows
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.bar(
        np.arange(len(values)),
        values,
        yerr=errors,
        capsize=4
    )

    ax.set_xticks(
        np.arange(len(values))
    )

    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$S=\sum w_{\mathrm{lumi}}$")
    ax.set_title("ttH signal yield in SR")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        outdir / "ttH_signal_yield_by_fold_and_total.png",
        dpi=170
    )

    plt.close(fig)


def plot_s_and_b(rows, total, outdir):

    all_rows = rows + [total]

    labels = [
        f"Fold {r['fold']}"
        for r in rows
    ] + ["Combined"]

    x = np.arange(len(all_rows))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5)
    )

    # Signal
    axes[0].bar(
        x,
        [r["S_yield"] for r in all_rows],
        yerr=[r["S_stat_err"] for r in all_rows],
        capsize=3
    )

    axes[0].set_title("ttH signal")
    axes[0].set_ylabel("Weighted yield")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        labels,
        rotation=25
    )

    axes[0].grid(
        axis="y",
        alpha=0.25
    )

    # Background
    axes[1].bar(
        x,
        [r["B_estimate"] for r in all_rows],
        yerr=[r["B_stat_err"] for r in all_rows],
        capsize=3
    )

    axes[1].set_title("CR-estimated background")
    axes[1].set_ylabel("Predicted yield")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        labels,
        rotation=25
    )

    axes[1].grid(
        axis="y",
        alpha=0.25
    )

    fig.suptitle(
        "Inclusive SR signal and background yields"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "signal_background_yields.png",
        dpi=170
    )

    plt.close(fig)


# =============================================================================
# 10. Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default=str(
            REPO / "tools" /
            "results_cr_alpha_signal_yield"
        )
    )

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    missing_files = []
    fold_rows = []

    print("=" * 100)
    print("INCLUSIVE CR ALPHA + ttH SIGNAL YIELD")
    print("=" * 100)

    print(f"Nominal Data: {DATA_BASE}")
    print(f"CR Data     : {CR_BASE}")
    print(f"ttH MC      : {MC_BASE}")

    print("\nSelection (pure mass window, no tag / is_SR restriction):")
    print("  Data SB : m_llg in 110-120 + 130-180 GeV")
    print("  CR SB   : m_llg in 110-120 + 130-180 GeV")
    print("  CR SR   : m_llg in 120-130 GeV")
    print("  Signal  : ttH, m_llg in 120-130 GeV")
    print("  Fold    : photon_eta_fold % 4 == model fold")

    # -------------------------------------------------------------------------
    # Four folds
    # -------------------------------------------------------------------------

    for fold in range(4):

        print(f"\nLoading fold {fold} ...")

        nominal_all = load_data_population(
            fold,
            "nominal",
            missing_files
        )

        cr_all = load_data_population(
            fold,
            "cr",
            missing_files
        )

        signal_all = load_tth_signal(
            fold,
            missing_files
        )

        # Each model uses its own test parity
        nominal_test = select_parity(
            nominal_all,
            fold
        )

        cr_test = select_parity(
            cr_all,
            fold
        )

        signal_test = select_parity(
            signal_all,
            fold
        )

        row = evaluate_fold(
            fold,
            nominal_test,
            cr_test,
            signal_test
        )

        fold_rows.append(row)

        print(
            f"  Data SB = {row['N_Data_SB']}"
        )

        print(
            f"  CR SB   = {row['N_CR_SB']}"
        )

        print(
            f"  CR SR   = {row['N_CR_SR']}"
        )

        print(
            f"  alpha   = {fmt(row['alpha'], 6)}"
        )

        print(
            f"  B       = {fmt(row['B_estimate'], 5)}"
        )

        print(
            f"  ttH N   = {row['N_S_events']}"
        )

        print(
            f"  S       = {row['S_yield']:.6f}"
            f" +/- {row['S_stat_err']:.6f}"
        )

        print(
            f"  Neff(S) = {row['S_Neff']:.1f}"
        )

    # -------------------------------------------------------------------------
    # Combined result
    # -------------------------------------------------------------------------

    total = make_total_row(fold_rows)

    print_results(
        fold_rows,
        total
    )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    write_csv(
        outdir / "cr_alpha_signal_yield_by_fold.csv",
        fold_rows
    )

    write_csv(
        outdir / "cr_alpha_signal_yield_total.csv",
        [total]
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    plot_event_counts(
        fold_rows,
        outdir
    )

    plot_alpha(
        fold_rows,
        total,
        outdir
    )

    plot_b(
        fold_rows,
        total,
        outdir
    )

    plot_signal_yield(
        fold_rows,
        total,
        outdir
    )

    plot_s_and_b(
        fold_rows,
        total,
        outdir
    )

    # -------------------------------------------------------------------------
    # Missing files
    # -------------------------------------------------------------------------

    if missing_files:

        missing_path = (
            outdir / "missing_files.txt"
        )

        missing_path.write_text(
            "\n".join(sorted(set(missing_files))) + "\n",
            encoding="utf-8"
        )

        print(
            f"\nWARNING: {len(set(missing_files))} "
            f"input files missing."
        )

        print(f"See: {missing_path}")

    else:

        print("\nAll expected era files were found.")

    print("\nOutput directory:")
    print(outdir)

    print("\nDone. No significance Z calculated.")


if __name__ == "__main__":
    main()