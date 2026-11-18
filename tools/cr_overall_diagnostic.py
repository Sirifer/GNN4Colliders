#!/usr/bin/env python3
"""
Overall CR diagnostic for the v3 4-class, 4-fold ttH analysis.

This version deliberately does NOT split events into Low/Medium/High score
regions and does NOT split the five analysis categories. It studies the
inclusive samples first.

For each model fold k, it uses only TEST parity k:
    fold 0 -> photon_eta_fold % 4 == 0
    fold 1 -> photon_eta_fold % 4 == 1
    fold 2 -> photon_eta_fold % 4 == 2
    fold 3 -> photon_eta_fold % 4 == 3

Samples counted and plotted:
    Nominal Data SB: 110 <= m_llg < 120 or 130 <= m_llg <= 180
    CR Data SR:      120 < m_llg < 130
    CR Data SB:      same two sidebands as Nominal Data

The score is ttH_gnn_score[1] (class index 1). Verify the class ordering
against the training label mapping.

Outputs:
    event_counts_by_fold.csv
    score_summary.csv
    score_distribution_counts.png
    score_distribution_normalized.png
    score_distribution_sb_split.png
    score_distribution_by_fold.png

Run:
    python cr_overall_diagnostic.py

Optional:
    python cr_overall_diagnostic.py --output-dir tools/results_cr_overall
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
# 1. Configuration -- aligned with significance_4class_v3_cr.py
# =============================================================================

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")

DATA_BASE = (
    REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"
)
CR_BASE = (
    REPO / "root_p2root_v3_cr_pt_only_scored_ttH_4class_4fold_fold"
)

TREE_NOM = "merged_SR_Sideband"
TREE_CR = "merged_CR_lowIDMVA"

MASS = "m_llg"
SCORE = "ttH_gnn_score"
FOLDVAR = "photon_eta_fold"

ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]

MASS_SR = lambda m: (m > 120.0) & (m < 130.0)
MASS_SB_LOW = lambda m: (m >= 110.0) & (m < 120.0)
MASS_SB_HIGH = lambda m: (m >= 130.0) & (m <= 180.0)
MASS_SB = lambda m: MASS_SB_LOW(m) | MASS_SB_HIGH(m)

SCORE_BINS = np.linspace(0.0, 1.0, 51)


# =============================================================================
# 2. ROOT reading
# =============================================================================

def extract_tth_score(raw):
    """Convert vector<float> scores to (N,4), then return column 1."""
    raw = np.asarray(raw)

    if raw.dtype == object or raw.ndim == 1:
        rows = [np.asarray(x, dtype=np.float32).reshape(-1) for x in raw]
        bad = [i for i, x in enumerate(rows) if x.size != 4]
        if bad:
            raise ValueError(
                f"{SCORE}: expected 4 values/event; "
                f"{len(bad)} malformed entries, first indices={bad[:10]}"
            )
        if not rows:
            return np.empty(0, dtype=np.float32)
        matrix = np.vstack(rows)
    else:
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 4:
            raise ValueError(
                f"{SCORE}: expected array shape (N,4), got {matrix.shape}"
            )

    return matrix[:, 1].astype(np.float64)


def read_fold_sample(model_fold, sample, missing_files):
    """
    Read all eras from the scored directory for one model fold.
    Return arrays for the physical mass windows, before parity selection.
    """
    if sample == "nominal":
        base = DATA_BASE.parent / f"{DATA_BASE.name}{model_fold}"
        tree_name = TREE_NOM
    elif sample == "cr":
        base = CR_BASE.parent / f"{CR_BASE.name}{model_fold}"
        tree_name = TREE_CR
    else:
        raise ValueError(sample)

    parts = []

    for era in ERAS:
        path = base / "Data" / "Data" / f"{era}.root"
        if not path.exists():
            missing_files.append(str(path))
            continue

        with uproot.open(path) as root_file:
            if tree_name not in root_file:
                raise KeyError(
                    f"TTree '{tree_name}' not found in {path}; "
                    f"available keys={root_file.keys()}"
                )

            tree = root_file[tree_name]
            required = [MASS, SCORE, FOLDVAR]
            absent = [name for name in required if name not in tree]
            if absent:
                raise KeyError(f"Missing branches {absent} in {path}")

            arrays = tree.arrays(required, library="np")

        mass = np.asarray(arrays[MASS], dtype=np.float64)
        score = extract_tth_score(arrays[SCORE])
        parity = np.asarray(arrays[FOLDVAR], dtype=np.int32) % 4

        if sample == "nominal":
            # Only nominal Data SB is used in the CR transfer factor.
            keep = np.isfinite(mass) & MASS_SB(mass)
        else:
            # Keep CR SR and both CR sidebands.
            keep = np.isfinite(mass) & (MASS_SR(mass) | MASS_SB(mass))

        if np.any(keep):
            parts.append({
                "mass": mass[keep],
                "score": score[keep],
                "parity": parity[keep],
            })

    if not parts:
        return {
            "mass": np.array([], dtype=np.float64),
            "score": np.array([], dtype=np.float64),
            "parity": np.array([], dtype=np.int32),
        }

    return {
        key: np.concatenate([part[key] for part in parts])
        for key in ("mass", "score", "parity")
    }


def select_parity(population, parity):
    mask = population["parity"] == parity
    return {
        key: value[mask]
        for key, value in population.items()
    }


# =============================================================================
# 3. Build inclusive OOF samples
# =============================================================================

def empty_score_list():
    return {
        "nominal_sb": [],
        "nominal_sb_low": [],
        "nominal_sb_high": [],
        "cr_sr": [],
        "cr_sb": [],
        "cr_sb_low": [],
        "cr_sb_high": [],
    }


def append_sample_scores(score_store, nominal, cr):
    nm = nominal["mass"]
    ns = nominal["score"]
    cm = cr["mass"]
    cs = cr["score"]

    masks = {
        "nominal_sb": MASS_SB(nm),
        "nominal_sb_low": MASS_SB_LOW(nm),
        "nominal_sb_high": MASS_SB_HIGH(nm),
        "cr_sr": MASS_SR(cm),
        "cr_sb": MASS_SB(cm),
        "cr_sb_low": MASS_SB_LOW(cm),
        "cr_sb_high": MASS_SB_HIGH(cm),
    }

    score_store["nominal_sb"].append(ns[masks["nominal_sb"]])
    score_store["nominal_sb_low"].append(ns[masks["nominal_sb_low"]])
    score_store["nominal_sb_high"].append(ns[masks["nominal_sb_high"]])
    score_store["cr_sr"].append(cs[masks["cr_sr"]])
    score_store["cr_sb"].append(cs[masks["cr_sb"]])
    score_store["cr_sb_low"].append(cs[masks["cr_sb_low"]])
    score_store["cr_sb_high"].append(cs[masks["cr_sb_high"]])


def concatenate_score_store(store):
    return {
        key: (
            np.concatenate(parts)
            if parts and any(len(x) for x in parts)
            else np.array([], dtype=np.float64)
        )
        for key, parts in store.items()
    }


# =============================================================================
# 4. Counts and score summaries
# =============================================================================

def finite_scores(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def score_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    finite = finite_scores(values)

    if finite.size == 0:
        return {
            "N": int(values.size),
            "N_finite": 0,
            "N_nonfinite": int(values.size),
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "N_outside_0_1": 0,
        }

    q10, q25, median, q75, q90 = np.quantile(
        finite, [0.10, 0.25, 0.50, 0.75, 0.90]
    )

    return {
        "N": int(values.size),
        "N_finite": int(finite.size),
        "N_nonfinite": int(values.size - finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(median),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "N_outside_0_1": int(np.count_nonzero((finite < 0) | (finite > 1))),
    }


def build_count_row(fold, nominal, cr):
    nm = nominal["mass"]
    cm = cr["mass"]

    return {
        "fold": fold,
        "test_parity": fold,
        "N_Nominal_Data_SB_low": int(np.count_nonzero(MASS_SB_LOW(nm))),
        "N_Nominal_Data_SB_high": int(np.count_nonzero(MASS_SB_HIGH(nm))),
        "N_Nominal_Data_SB_total": int(np.count_nonzero(MASS_SB(nm))),
        "N_CR_SR": int(np.count_nonzero(MASS_SR(cm))),
        "N_CR_SB_low": int(np.count_nonzero(MASS_SB_LOW(cm))),
        "N_CR_SB_high": int(np.count_nonzero(MASS_SB_HIGH(cm))),
        "N_CR_SB_total": int(np.count_nonzero(MASS_SB(cm))),
        "N_CR_SR_plus_SB": int(np.count_nonzero(MASS_SR(cm) | MASS_SB(cm))),
    }


def make_summary_rows(score_store_by_fold, score_store_total):
    rows = []

    labels = {
        "nominal_sb": "Nominal Data SB (combined)",
        "nominal_sb_low": "Nominal Data SB (110-120)",
        "nominal_sb_high": "Nominal Data SB (130-180)",
        "cr_sr": "CR Data SR",
        "cr_sb": "CR Data SB (combined)",
        "cr_sb_low": "CR Data SB (110-120)",
        "cr_sb_high": "CR Data SB (130-180)",
    }

    for fold, store in score_store_by_fold.items():
        for key, label in labels.items():
            rows.append({
                "sample": label,
                "fold": fold,
                **score_statistics(store[key]),
            })

    for key, label in labels.items():
        rows.append({
            "sample": label,
            "fold": "ALL_TEST_PARITIES",
            **score_statistics(score_store_total[key]),
        })

    return rows


def print_counts(rows):
    print("\n" + "=" * 112)
    print("INCLUSIVE EVENT COUNTS -- NO SCORE BINNING")
    print("=" * 112)
    header = (
        f"{'Fold':>6} {'Data SB low':>13} {'Data SB high':>14} {'Data SB total':>14}"
        f" {'CR SR':>9} {'CR SB low':>11} {'CR SB high':>12} {'CR SB total':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['fold']:>6}"
            f"{row['N_Nominal_Data_SB_low']:13d}"
            f"{row['N_Nominal_Data_SB_high']:14d}"
            f"{row['N_Nominal_Data_SB_total']:14d}"
            f"{row['N_CR_SR']:9d}"
            f"{row['N_CR_SB_low']:11d}"
            f"{row['N_CR_SB_high']:12d}"
            f"{row['N_CR_SB_total']:12d}"
        )

    total = {
        key: sum(row[key] for row in rows)
        for key in rows[0]
        if key.startswith("N_")
    }
    print("-" * len(header))
    print(
        f"{'TOTAL':>6}"
        f"{total['N_Nominal_Data_SB_low']:13d}"
        f"{total['N_Nominal_Data_SB_high']:14d}"
        f"{total['N_Nominal_Data_SB_total']:14d}"
        f"{total['N_CR_SR']:9d}"
        f"{total['N_CR_SB_low']:11d}"
        f"{total['N_CR_SB_high']:12d}"
        f"{total['N_CR_SB_total']:12d}"
    )


def print_score_summary(rows):
    print("\n" + "=" * 112)
    print("INCLUSIVE ttH SCORE SUMMARY")
    print("=" * 112)
    print(
        f"{'Sample':34s} {'Fold':>16s} {'N':>9s} {'Mean':>9s} "
        f"{'Std':>9s} {'Median':>9s} {'Q10':>9s} {'Q90':>9s}"
    )
    print("-" * 112)

    for row in rows:
        print(
            f"{row['sample']:34s} {str(row['fold']):>16s}"
            f"{row['N']:9d}"
            f"{row['mean']:9.4f}" if np.isfinite(row["mean"]) else
            f"{row['sample']:34s} {str(row['fold']):>16s}{row['N']:9d}"
        )
        if np.isfinite(row["mean"]):
            print(
                f"{'':34s} {'':16s}{'':9s}"
                f"{row['mean']:9.4f}{row['std']:9.4f}"
                f"{row['median']:9.4f}{row['q10']:9.4f}{row['q90']:9.4f}"
            )


# =============================================================================
# 5. Plots
# =============================================================================

def plot_overall_distributions(score_store, output_dir):
    # Plot 1: raw event counts for the three populations used by the method.
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_items = [
        ("nominal_sb", "Nominal Data SB"),
        ("cr_sb", "CR Data SB"),
        ("cr_sr", "CR Data SR"),
    ]

    for key, label in plot_items:
        values = finite_scores(score_store[key])
        ax.hist(
            values, bins=SCORE_BINS, histtype="step", linewidth=1.8,
            label=f"{label} (N={len(values)})",
        )

    ax.set_xlabel(r"$p_{\mathrm{ttH}}$ score")
    ax.set_ylabel("Events")
    ax.set_title("Inclusive ttH-score distributions (event counts)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_counts.png", dpi=160)
    plt.close(fig)

    # Plot 2: unit-normalised shapes for direct shape comparison.
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label in plot_items:
        values = finite_scores(score_store[key])
        if values.size == 0:
            continue
        ax.hist(
            values, bins=SCORE_BINS, density=True,
            histtype="step", linewidth=1.8, label=label,
        )

    ax.set_xlabel(r"$p_{\mathrm{ttH}}$ score")
    ax.set_ylabel("Normalised density")
    ax.set_title("Inclusive ttH-score shapes (unit normalised)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_normalized.png", dpi=160)
    plt.close(fig)

    # Plot 3: sideband components separately, to inspect mass dependence.
    fig, ax = plt.subplots(figsize=(9, 6))
    split_items = [
        ("nominal_sb_low", "Nominal Data 110-120"),
        ("nominal_sb_high", "Nominal Data 130-180"),
        ("cr_sb_low", "CR Data 110-120"),
        ("cr_sb_high", "CR Data 130-180"),
        ("cr_sr", "CR Data 120-130"),
    ]

    for key, label in split_items:
        values = finite_scores(score_store[key])
        if values.size == 0:
            continue
        ax.hist(
            values, bins=SCORE_BINS, density=True,
            histtype="step", linewidth=1.6, label=f"{label} (N={len(values)})",
        )

    ax.set_xlabel(r"$p_{\mathrm{ttH}}$ score")
    ax.set_ylabel("Normalised density")
    ax.set_title("ttH-score shapes: low/high sidebands and CR SR")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_sb_split.png", dpi=160)
    plt.close(fig)


def plot_by_fold(score_store_by_fold, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for fold, ax in enumerate(axes):
        store = score_store_by_fold[fold]
        items = [
            ("nominal_sb", "Nominal Data SB"),
            ("cr_sb", "CR Data SB"),
            ("cr_sr", "CR Data SR"),
        ]

        for key, label in items:
            values = finite_scores(store[key])
            if values.size == 0:
                continue
            ax.hist(
                values, bins=SCORE_BINS, density=True,
                histtype="step", linewidth=1.5, label=f"{label} (N={len(values)})",
            )

        ax.set_title(f"Fold {fold} (test parity {fold})")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$p_{\mathrm{ttH}}$")
        ax.set_ylabel("Normalised density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Inclusive ttH-score distributions by test fold")
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_by_fold.png", dpi=160)
    plt.close(fig)


# =============================================================================
# 6. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "tools" / "results_cr_overall_diagnostic"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_files = []
    count_rows = []
    score_store_by_fold = {}
    score_store_total = empty_score_list()

    print("=" * 100)
    print("OVERALL CR DIAGNOSTIC -- NO LOW/MEDIUM/HIGH SCORE BINNING")
    print("=" * 100)
    print(f"Nominal Data base: {DATA_BASE}")
    print(f"CR Data base     : {CR_BASE}")
    print(f"Output directory : {output_dir}")
    print("Score used       : ttH_gnn_score[1] (verify class index 1 = ttH)")
    print("Fold selection   : model fold k uses photon_eta_fold % 4 == k")
    print("No category or score-region splitting is applied.")

    for fold in range(4):
        print("\n" + "#" * 100)
        print(f"MODEL FOLD {fold}: TEST PARITY = {fold}")
        print("#" * 100)

        nominal_all = read_fold_sample(fold, "nominal", missing_files)
        cr_all = read_fold_sample(fold, "cr", missing_files)

        nominal_test = select_parity(nominal_all, fold)
        cr_test = select_parity(cr_all, fold)

        count_row = build_count_row(fold, nominal_test, cr_test)
        count_rows.append(count_row)

        score_store = empty_score_list()
        append_sample_scores(score_store, nominal_test, cr_test)
        score_store = concatenate_score_store(score_store)
        score_store_by_fold[fold] = score_store

        # Accumulate the unique test-parity samples across the four folds.
        for key in score_store_total:
            score_store_total[key].append(score_store[key])

        print(
            f"Nominal Data SB: {count_row['N_Nominal_Data_SB_total']} "
            f"(low={count_row['N_Nominal_Data_SB_low']}, "
            f"high={count_row['N_Nominal_Data_SB_high']})"
        )
        print(
            f"CR Data: SR={count_row['N_CR_SR']}, "
            f"SB={count_row['N_CR_SB_total']} "
            f"(low={count_row['N_CR_SB_low']}, "
            f"high={count_row['N_CR_SB_high']})"
        )
        for label, key in [
            ("Nominal Data SB", "nominal_sb"),
            ("CR Data SB", "cr_sb"),
            ("CR Data SR", "cr_sr"),
        ]:
            stats = score_statistics(score_store[key])
            print(
                f"{label:18s}: N={stats['N']:7d}, "
                f"mean={stats['mean']:.5f}, std={stats['std']:.5f}, "
                f"median={stats['median']:.5f}"
            )

    score_store_total = concatenate_score_store(score_store_total)

    print_counts(count_rows)

    summary_rows = make_summary_rows(
        score_store_by_fold, score_store_total
    )
    print_score_summary(summary_rows)

    # Save event counts by fold and total.
    count_csv = output_dir / "event_counts_by_fold.csv"
    with open(count_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0].keys()))
        writer.writeheader()
        writer.writerows(count_rows)

        total = {
            "fold": "TOTAL",
            "test_parity": "0+1+2+3",
        }
        for key in count_rows[0]:
            if key.startswith("N_"):
                total[key] = sum(row[key] for row in count_rows)
        writer.writerow(total)

    summary_csv = output_dir / "score_summary.csv"
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_overall_distributions(score_store_total, output_dir)
    plot_by_fold(score_store_by_fold, output_dir)

    if missing_files:
        missing_path = output_dir / "missing_files.txt"
        missing_path.write_text(
            "\n".join(sorted(set(missing_files))) + "\n",
            encoding="utf-8",
        )
        print(
            f"\nWARNING: {len(set(missing_files))} input files missing; "
            f"see {missing_path}"
        )
    else:
        print("\nAll expected era files were found.")

    print("\nOutputs:")
    print(f"  {count_csv}")
    print(f"  {summary_csv}")
    for filename in [
        "score_distribution_counts.png",
        "score_distribution_normalized.png",
        "score_distribution_sb_split.png",
        "score_distribution_by_fold.png",
    ]:
        print(f"  {output_dir / filename}")
    print("\nDone.")


if __name__ == "__main__":
    main()
