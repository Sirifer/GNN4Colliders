#!/usr/bin/env python3
"""Rewrite the broken `photon_eta_fold` branch of the v2 Data samples.

Why this exists
---------------
The v2 MC files carry

    photon_eta_fold = int32(photon_eta * 10000)

which is what `Datasets[...].folding` / the analysis use as the fold variable
(`fold = photon_eta_fold % 4`). Verified on
`root_p2root_v2_pt_only/Signal/ttH/2022preEE.root`: the stored branch matches
`int32(photon_eta*10000)` for 9346 / 9346 events and covers all four parities.

The v2 **Data** files do not. Their stored `photon_eta_fold` only ever contains
the values 0 and 1, e.g.

    2022preEE.root  n=224   stored uniq [0 1]  parity counts [102 122 0 0]
    2024.root       n=4346  stored uniq [0 1]  parity counts [2170 2176 0 0]

while `int32(photon_eta*10000)` on the very same events gives

    2022preEE.root  parity counts [42 48 60 74]
    2024.root       parity counts [1079 1081 1091 1095]

and matches the stored branch in 0 / 224 and 0 / 4346 events. So the branch is
not a different convention, it is simply wrong: only two of the four folds can
be selected from it, which is why the v2 Data side could not be split in four.

`photon_eta` itself is present and correct in the Data files, so the fold
variable is recoverable without re-running anything upstream.

What it does
------------
For every era of `root_p2root_v2_pt_only_data/` it copies the tree
`merged_SR_Sideband` (all branches unchanged) into

    root_p2root_v2_pt_only_data_foldfix/<era>.root

with `photon_eta_fold` replaced by `int32(photon_eta*10000)`. The copy is
verified before being accepted: same number of entries, every other branch
identical, four parities present, and the SR/SB event counts unchanged.

The scored Data products are then produced from this directory with the
existing four jobs `hzg_ttH_4class_on_v2_4fold_data_fix_fold{k}_inference.sbatch`
(see `tools/make_v2_4class_on_v2_jobs.py --data-fix`), and the significance
script reads them with `--data-dir`.

Usage:
    python tools/fix_v2_data_fold.py --check     # report only
    python tools/fix_v2_data_fold.py             # write the fixed copies
"""

import argparse
from pathlib import Path

import numpy as np
import uproot

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")

SRC_DIR = REPO / "root_p2root_v2_pt_only_data"
DST_DIR = REPO / "root_p2root_v2_pt_only_data_foldfix"

TREE = "merged_SR_Sideband"
FOLDVAR = "photon_eta_fold"
PHOTON_ETA = "photon_eta"
TAG = "source_tree_tag"
NFOLD = 4

# Same six eras as the v2 MC samples and the existing v2 Data scoring.
# `root_p2root_v2_pt_only_data` also holds 2026.root, which is not in the v2
# MC file lists; it is left out so the two sides stay on identical eras.
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix",
        "2024", "2025"]


def compute_fold(photon_eta):
    """The v2-MC definition of the fold variable."""
    return np.asarray(photon_eta * 10000.0, dtype=np.float64).astype(np.int32)


def parity_counts(foldvar):
    counts = np.zeros(NFOLD, dtype=np.int64)
    rem = np.mod(np.asarray(foldvar, dtype=np.int64), NFOLD)
    for p in range(NFOLD):
        counts[p] = int(np.count_nonzero(rem == p))
    return counts


def sb_counts(foldvar, tag):
    """Sideband events (odd source_tree_tag) per parity."""
    sb = (np.asarray(tag, dtype=np.int64) % 2) == 1
    rem = np.mod(np.asarray(foldvar, dtype=np.int64), NFOLD)
    return np.array([int(np.count_nonzero(sb & (rem == p)))
                     for p in range(NFOLD)], dtype=np.int64)


def describe(counts):
    return "[" + " ".join(str(int(c)) for c in counts) + "]"


def process_era(era, check_only):
    src = SRC_DIR / f"{era}.root"
    if not src.exists():
        return None, f"missing source file {src}"

    with uproot.open(str(src)) as f:
        if TREE not in f:
            return None, f"{src}: TTree {TREE!r} not found"
        tree = f[TREE]
        branches = list(tree.keys())
        if FOLDVAR not in branches or PHOTON_ETA not in branches:
            return None, (f"{src}: needs both {FOLDVAR!r} and "
                          f"{PHOTON_ETA!r}")
        arrays = tree.arrays(library="np")

    n = len(arrays[FOLDVAR])
    old_fold = np.asarray(arrays[FOLDVAR])
    new_fold = compute_fold(arrays[PHOTON_ETA])
    tag = np.asarray(arrays[TAG])

    lines = [f"--- {era}.root  ({n} events, {len(branches)} branches)"]
    lines.append(f"   stored {FOLDVAR} uniq (first 8): "
                 f"{np.unique(old_fold)[:8].tolist()}")
    lines.append(f"   stored   parity counts: {describe(parity_counts(old_fold))}")
    lines.append(f"   recomputed parity counts: "
                 f"{describe(parity_counts(new_fold))}")
    lines.append(f"   stored SB per parity: {describe(sb_counts(old_fold, tag))}")
    lines.append(f"   fixed  SB per parity: "
                 f"{describe(sb_counts(new_fold, tag))}")
    n_match = int(np.count_nonzero(old_fold == new_fold))
    lines.append(f"   stored == int32(photon_eta*10000): {n_match} / {n}")

    if check_only:
        return new_fold, "\n".join(lines)

    # Keep the old branch name, replace its content.
    expected = dict(arrays)
    replaced = dict(arrays)
    replaced[FOLDVAR] = new_fold.astype(np.int32)

    dst = DST_DIR / f"{era}.root"
    DST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".root.tmp")
    with uproot.recreate(str(tmp)) as fout:
        fout[TREE] = replaced

    # --- verify the copy before accepting it -----------------------------
    with uproot.open(str(tmp)) as f:
        if TREE not in f:
            raise RuntimeError(f"{tmp}: TTree {TREE!r} missing in the copy")
        back = f[TREE].arrays(library="np")

    if len(back[FOLDVAR]) != n:
        raise RuntimeError(f"{tmp}: entry count changed "
                           f"{n} -> {len(back[FOLDVAR])}")
    if set(back.keys()) != set(branches):
        raise RuntimeError(f"{tmp}: branch list changed")

    for k in branches:
        a = back[k]
        if k == FOLDVAR:
            continue
        if not np.array_equal(np.asarray(a), np.asarray(expected[k])):
            raise RuntimeError(f"{tmp}: branch {k!r} differs from the source")
    if not np.array_equal(np.asarray(back[FOLDVAR], dtype=np.int32), new_fold):
        raise RuntimeError(f"{tmp}: {FOLDVAR!r} was not written correctly")

    got = parity_counts(back[FOLDVAR])
    if int(np.count_nonzero(got)) != NFOLD:
        raise RuntimeError(f"{tmp}: only parities "
                           f"{np.nonzero(got)[0].tolist()} survive")

    tmp.replace(dst)
    lines.append(f"   wrote {dst}")
    return new_fold, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute photon_eta_fold for the v2 Data samples.")
    parser.add_argument("--check", action="store_true",
                        help="report only, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing fixed files")
    args = parser.parse_args()

    print(f"source: {SRC_DIR}")
    print(f"target: {DST_DIR}")
    print()

    report = []
    total_sb = np.zeros(NFOLD, dtype=np.int64)

    for era in ERAS:
        dst = DST_DIR / f"{era}.root"
        if dst.exists() and not args.force and not args.check:
            print(f"[skip] exists: {dst.name} (use --force)")
            continue
        fold, text = process_era(era, args.check)
        print(text)
        report.append(text)
        if fold is not None:
            with uproot.open(str(SRC_DIR / f"{era}.root")) as f:
                tag = np.asarray(
                    f[TREE].arrays([TAG], library="np")[TAG])
            total_sb += sb_counts(fold, tag)
        print()

    print(f"Data sideband events per fold parity (all six eras, fixed): "
          f"{describe(total_sb)}")
    print(f"total Data sideband events: {int(total_sb.sum())}")
    print("expected y-axis value: alphas of the four folds should now all be "
          "~N(Data SB)/Y(Bkg SB), instead of only fold 0/1 having a value.")

    if not args.check:
        DST_DIR.mkdir(parents=True, exist_ok=True)
        (DST_DIR / "foldfix_report.txt").write_text(
            "v2 Data photon_eta_fold rewrite\n"
            "photon_eta_fold := int32(photon_eta * 10000)\n\n"
            + "\n\n".join(report)
            + f"\n\nData sideband per parity: {describe(total_sb)}\n",
            encoding="utf-8")
        print(f"wrote {DST_DIR / 'foldfix_report.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
