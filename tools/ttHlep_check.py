#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict

import numpy as np
import uproot


# =============================================================================
# Config
# =============================================================================

DATA_DIR = Path(
    "/scratchfs2/cms/huangsuyun/"
    "GNN4Colliders/root_p2root_v2_scored"
)

TREE = "merged_SR_Sideband"

WEIGHT = "weight_lumi"
MASS = "m_llg"
SCORE = "ttH_gnn_score"
IS_SR = "is_SR"
TREE_TAG = "source_tree_tag"
EVENT = "event"

# ttHlep
TTHLEP_TAG = 2

MASS_LO = 120.0
MASS_HI = 130.0

TOP_N = 30


# =============================================================================
# Run3
# =============================================================================

RUN3 = [
    "2022preEE",
    "2022postEE",
    "2023preBPix",
    "2023postBPix",
    "2024",
    "2025",
]


# =============================================================================
# Background samples
# =============================================================================

BACKGROUND = [
    ("Signal/VH",                 RUN3),
    ("Signal/ggF",                RUN3),
    ("Signal/VBF",                RUN3),

    ("Bkg_MC/DYJetsToLL_Stitch", RUN3),
    ("Bkg_MC/TT",                 RUN3),
    ("Bkg_MC/DYGto2LG_10to100",  RUN3),
    ("Bkg_MC/ZG2JToG2L2J",       RUN3),

    ("Bkg_MC/TWminusto2L2Nu",    RUN3),
    ("Bkg_MC/TbarWplusto2L2Nu",  RUN3),

    ("Bkg_MC/TTto2L2Nu",         RUN3),

    ("Bkg_MC/TTG1Jets_10to100",  RUN3),
    ("Bkg_MC/TTG1Jets_100to200", RUN3),
    ("Bkg_MC/TTG1Jets_200",      RUN3),

    ("Bkg_MC/WZ",                 RUN3),
    ("Bkg_MC/ZZ",                 RUN3),
    ("Bkg_MC/WW",                 RUN3),

    ("Bkg_MC/DYJetsToLL",         RUN3),
]


# =============================================================================
# Load ttHlep background events
# =============================================================================

def load_background():

    events = []

    branches = [
        WEIGHT,
        MASS,
        SCORE,
        IS_SR,
        TREE_TAG,
        EVENT,
    ]

    for sample, years in BACKGROUND:

        for year in years:

            filename = (
                DATA_DIR
                / sample
                / f"{year}.root"
            )

            if not filename.exists():
                continue

            try:

                with uproot.open(filename) as f:

                    if TREE not in f:
                        continue

                    arr = f[TREE].arrays(
                        branches,
                        library="np",
                    )

            except Exception as e:

                print(
                    f"WARNING: cannot read "
                    f"{filename}: {e}"
                )
                continue


            mask = (
                (arr[IS_SR] == 1)

                & (arr[MASS] > MASS_LO)
                & (arr[MASS] < MASS_HI)

                & (arr[TREE_TAG] == TTHLEP_TAG)

                & np.isfinite(arr[WEIGHT])
                & np.isfinite(arr[SCORE])
            )


            for i in np.where(mask)[0]:

                events.append({
                    "sample": sample,
                    "year": year,
                    "event": int(arr[EVENT][i]),
                    "weight": float(arr[WEIGHT][i]),
                    "score": float(arr[SCORE][i]),
                    "m_llg": float(arr[MASS][i]),
                })


    return events


# =============================================================================
# Overall summary
# =============================================================================

def print_weight_summary(title, events):

    weights = np.array(
        [x["weight"] for x in events]
    )

    positive = weights[weights > 0]
    negative = weights[weights < 0]

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(f"N total        = {len(weights)}")
    print(f"N positive     = {len(positive)}")
    print(f"N negative     = {len(negative)}")
    print()

    print(f"sum positive   = {np.sum(positive): .6f}")
    print(f"sum negative   = {np.sum(negative): .6f}")
    print("-" * 40)
    print(f"B = sum(weight)= {np.sum(weights): .6f}")

    if len(weights):
        print()
        print(f"most negative  = {np.min(weights): .6f}")
        print(f"most positive  = {np.max(weights): .6f}")


# =============================================================================
# Contribution by sample
# =============================================================================

def print_sample_summary(title, events):

    groups = defaultdict(list)

    for ev in events:
        groups[ev["sample"]].append(
            ev["weight"]
        )

    rows = []

    for sample, weights in groups.items():

        weights = np.asarray(weights)

        neg = weights[weights < 0]
        pos = weights[weights > 0]

        rows.append({
            "sample": sample,
            "n": len(weights),
            "nneg": len(neg),
            "sum": np.sum(weights),
            "neg": np.sum(neg),
            "pos": np.sum(pos),
            "min": np.min(weights),
        })


    # Most negative sample first
    rows.sort(
        key=lambda x: x["neg"]
    )


    print()
    print("=" * 125)
    print(title)
    print("=" * 125)

    print(
        f"{'Sample':<32}"
        f"{'N':>9}"
        f"{'Nneg':>9}"
        f"{'sum neg':>15}"
        f"{'sum pos':>15}"
        f"{'B':>15}"
        f"{'min weight':>15}"
    )

    print("-" * 125)


    for r in rows:

        print(
            f"{r['sample']:<32}"
            f"{r['n']:>9d}"
            f"{r['nneg']:>9d}"
            f"{r['neg']:>15.6f}"
            f"{r['pos']:>15.6f}"
            f"{r['sum']:>15.6f}"
            f"{r['min']:>15.6f}"
        )


# =============================================================================
# Individual negative events
# =============================================================================

def print_most_negative(title, events, top_n=30):

    negative = [
        ev for ev in events
        if ev["weight"] < 0
    ]

    # Direct signed ordering:
    # -10, -5, -1, ...
    negative.sort(
        key=lambda x: x["weight"]
    )


    print()
    print("=" * 130)
    print(title)
    print("=" * 130)

    print(
        f"{'Rank':<6}"
        f"{'Sample':<32}"
        f"{'Year':<15}"
        f"{'Event':>18}"
        f"{'weight_lumi':>16}"
        f"{'GNN score':>14}"
        f"{'m_llg':>12}"
    )

    print("-" * 130)


    cumulative = 0.0

    for rank, ev in enumerate(
        negative[:top_n],
        start=1,
    ):

        cumulative += ev["weight"]

        print(
            f"{rank:<6}"
            f"{ev['sample']:<32}"
            f"{ev['year']:<15}"
            f"{ev['event']:>18d}"
            f"{ev['weight']:>16.6f}"
            f"{ev['score']:>14.6f}"
            f"{ev['m_llg']:>12.4f}"
        )

    print("-" * 130)

    print(
        f"Sum of TOP {min(top_n, len(negative))} "
        f"negative weights = {cumulative:.6f}"
    )


# =============================================================================
# Sample + year summary
# =============================================================================

def print_sample_year_summary(events):

    groups = defaultdict(list)

    for ev in events:

        key = (
            ev["sample"],
            ev["year"],
        )

        groups[key].append(
            ev["weight"]
        )


    rows = []

    for (sample, year), weights in groups.items():

        weights = np.asarray(weights)

        neg = weights[weights < 0]
        pos = weights[weights > 0]

        rows.append({
            "sample": sample,
            "year": year,
            "n": len(weights),
            "nneg": len(neg),
            "neg": np.sum(neg),
            "pos": np.sum(pos),
            "sum": np.sum(weights),
        })


    rows.sort(
        key=lambda x: x["neg"]
    )


    print()
    print("=" * 135)
    print("TEST BACKGROUND — contribution by SAMPLE and YEAR")
    print("=" * 135)

    print(
        f"{'Sample':<32}"
        f"{'Year':<15}"
        f"{'N':>9}"
        f"{'Nneg':>9}"
        f"{'sum neg':>15}"
        f"{'sum pos':>15}"
        f"{'B':>15}"
    )

    print("-" * 135)


    for r in rows:

        print(
            f"{r['sample']:<32}"
            f"{r['year']:<15}"
            f"{r['n']:>9d}"
            f"{r['nneg']:>9d}"
            f"{r['neg']:>15.6f}"
            f"{r['pos']:>15.6f}"
            f"{r['sum']:>15.6f}"
        )


# =============================================================================
# Main
# =============================================================================

def main():

    print("Loading ttHlep background events...")

    full = load_background()

    test = [
        ev
        for ev in full
        if ev["event"] % 4 == 0
    ]


    # -------------------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------------------

    print_weight_summary(
        "FULL ttHlep BACKGROUND",
        full,
    )

    print_weight_summary(
        "TEST ttHlep BACKGROUND — event % 4 == 0",
        test,
    )


    # -------------------------------------------------------------------------
    # Which samples cause it?
    # -------------------------------------------------------------------------

    print_sample_summary(
        "FULL BACKGROUND — contribution by sample",
        full,
    )

    print_sample_summary(
        "TEST BACKGROUND — contribution by sample",
        test,
    )


    # -------------------------------------------------------------------------
    # Which exact negative events?
    # -------------------------------------------------------------------------

    print_most_negative(
        "FULL — MOST NEGATIVE ttHlep BACKGROUND EVENTS",
        full,
        TOP_N,
    )

    print_most_negative(
        "TEST — MOST NEGATIVE ttHlep BACKGROUND EVENTS",
        test,
        TOP_N,
    )


    # -------------------------------------------------------------------------
    # Sample + year
    # -------------------------------------------------------------------------

    print_sample_year_summary(
        test
    )


    # -------------------------------------------------------------------------
    # FULL vs TEST
    # -------------------------------------------------------------------------

    B_full = sum(
        ev["weight"]
        for ev in full
    )

    B_test = sum(
        ev["weight"]
        for ev in test
    )

    print()
    print("=" * 80)
    print("FULL vs TEST")
    print("=" * 80)

    print(
        f"B_full          = {B_full:.6f}"
    )

    print(
        f"B_test          = {B_test:.6f}"
    )

    print(
        f"4 * B_test      = {4*B_test:.6f}"
    )

    print(
        f"B_full / 4      = {B_full/4:.6f}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()