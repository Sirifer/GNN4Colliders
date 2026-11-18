#!/usr/bin/env python3
"""
merge_trees_v3.py

Recursively copy + merge ROOT files from the v3 Mll80to100 sample into a
training-ready format, mirroring what merge_trees_all.py did for v2 but for
the new 4-class v3 setup.

What this does
--------------
For every selected sample (Signal ggF/ttH/VBF/VH, the 7 bkg samples, and Data)
and every era .root file it finds, the script merges the five analysis
categories' SR + Sideband trees into one TTree called "merged_SR_Sideband",
preserving the input directory structure (Signal/<sample>/<era>.root etc.).

Two extra columns are added to the merged tree:
    source_tree_tag  (int32)  -- which input tree the event came from
                                 0  SR__ttHhad       1  Sideband__ttHhad
                                 2  SR__ttHlep       3  Sideband__ttHlep
                                 4  SR__untag        5  Sideband__untag
                                 6  SR__VH3l         7  Sideband__VH3l
                                 8  SR__VHpTmiss     9  Sideband__VHpTmiss
    weight_lumi_abs  (float32) -- abs(weight_lumi), training weight
    photon_eta_fold  (int32)   -- int32(photon_eta * 10000); the fold variable.
    class_tag        (int32)   -- which 4-class the sample belongs to
                                  0  bkg        (TT + DYGto2LG_* + DYJetsToLL_Stitch)
                                  1  ttH
                                  2  VH
                                  3  ggF / VBF   (both signal samples merged into one class)
                                 -1  Data        (not a training class; Data is never trained)

class_tag is the same value for every event in a file (it is a property of the
sample, not the region). It mirrors the per-dataset `label` in the training
config, so downstream scripts can group events by class without re-deriving it
from the file path. Note the distinction from source_tree_tag, which is a
per-region tag (0-9) telling which of the 10 SR/Sideband trees an event came
from.

photon_eta_fold is computed here (int(photon_eta*10000) as int32) so that the
downstream dataset can partition events via `fold_var % n_folds`. This is the
same encoding produced by the old add_photon_eta_fold.py post-merge script,
folded into the merge step here to save a separate I/O pass.

weight_lumi_avg (the per-process constant mean weight) is NOT added here; it is
added by prepare_pt_only_inputs_v3.py in the next step, because it must be
computed across all era files of one process (ttH, VH, ...) at once.

Which samples are processed (whitelist)
---------------------------------------
Signal : ggF  ttH  VBF  VH
Bkg_MC : TT
         DYGto2LG_10to100  DYGto2LG_100to200  DYGto2LG_200to400
         DYGto2LG_400to600  DYGto2LG_600
         DYJetsToLL_Stitch
Data   : Data/Data   (all eras, 2016preVFP .. 2026)

Note: v3 files also contain SR__ggF / SR__VBF analysis-category trees and a
lot of *_JES_up / *_Electron_scale_down style systematic trees. These are NOT
part of the five training categories and are ignored: tree keys are matched
exactly against the 10 expected names via a set lookup, so the systematic
variants and the extra ggF/VBF category trees are never picked up.

Note: not every sample has every era (e.g. DYGto2LG_100to200 only has four
Run3 eras). The script walks the actual directory listing, so it only
processes eras that physically exist per sample -- nothing is hard-coded.

Usage:
    python merge_trees_v3.py
    python merge_trees_v3.py --force
    python merge_trees_v3.py --dry-run
    python merge_trees_v3.py --input-dir /path/to/root_p2root_v3_Mll80to100
                             --output-dir /path/to/root_p2root_v3_merged
    python merge_trees_v3.py --manifest merge_trees_v3_status.txt
"""

import argparse
import os
import sys
import tempfile
import time

import awkward as ak
import numpy as np
import uproot

BASE = "/publicfs/cms/user/laipeizhu/Eos/HZgamma"
REPO = "/scratchfs2/cms/huangsuyun/GNN4Colliders"
DEFAULT_INPUT_DIR = os.path.join(BASE, "root_p2root_v3_Mll80to100")
DEFAULT_OUTPUT_DIR = os.path.join(REPO, "root_p2root_v3_merged")
OUTPUT_TREE = "merged_SR_Sideband"

# Five training categories, identical to the v2 setup. ggF/VBF are signal
# SAMPLE names, not analysis categories -- their events still land in these
# five categories, so the category list is unchanged.
REGIONS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]
TREE_TAGS = []
for _i, _r in enumerate(REGIONS):
    TREE_TAGS.append((2 * _i, f"SR__{_r}"))
    TREE_TAGS.append((2 * _i + 1, f"Sideband__{_r}"))

# Whitelist of subdirectories (relative to <input-dir>) to process. Anything
# else under Bkg_MC (70+ samples, many irrelevant for the 4-class task) is
# skipped to keep the copy focused and fast.
SIGNAL_SAMPLES = ["ggF", "ttH", "VBF", "VH"]
BKG_SAMPLES = [
    "TT",
    "DYGto2LG_10to100",
    "DYGto2LG_100to200",
    "DYGto2LG_200to400",
    "DYGto2LG_400to600",
    "DYGto2LG_600",
    "DYJetsToLL_Stitch",
]

# class_tag mapping: which 4-class each sample belongs to. Mirrors the per-dataset
# `label` in the training config (bkg=0, ttH=1, VH=2, ggF+VBF=3). Data gets -1
# (Data is never trained). Keys are the sample subdirectory name (the leaf under
# Signal/ or Bkg_MC/); ggF and VBF share class 3.
SAMPLE_CLASS_TAG = {
    "ttH": 1,
    "VH": 2,
    "ggF": 3,
    "VBF": 3,
    "TT": 0,
    "DYGto2LG_10to100": 0,
    "DYGto2LG_100to200": 0,
    "DYGto2LG_200to400": 0,
    "DYGto2LG_400to600": 0,
    "DYGto2LG_600": 0,
    "DYJetsToLL_Stitch": 0,
}
DATA_CLASS_TAG = -1


def class_tag_for(input_path, input_dir):
    """Return the class_tag (int32) for a file from its sample dir name.

    The sample dir is the parent directory of the .root file relative to
    input_dir (e.g. 'Signal/ttH', 'Bkg_MC/TT', 'Data/Data'). Its leaf name is
    looked up in SAMPLE_CLASS_TAG; Data/Data maps to DATA_CLASS_TAG.
    """
    rel = os.path.relpath(input_path, input_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 2 and parts[0] == "Data" and parts[1] == "Data":
        return DATA_CLASS_TAG
    sample_name = parts[-2] if len(parts) >= 2 else None
    if sample_name not in SAMPLE_CLASS_TAG:
        raise ValueError(
            f"cannot determine class_tag for {input_path} "
            f"(sample dir name={sample_name!r} not in SAMPLE_CLASS_TAG)"
        )
    return SAMPLE_CLASS_TAG[sample_name]


def find_root_files(input_dir):
    """Find all .root files under the whitelisted Signal/Bkg_MC/Data dirs.

    Returns sorted list of absolute paths and the list of top-level subdirs
    actually scanned (for logging)."""
    files = []
    subdirs_to_scan = []

    for sample in SIGNAL_SAMPLES:
        p = os.path.join(input_dir, "Signal", sample)
        if os.path.isdir(p):
            subdirs_to_scan.append(os.path.join("Signal", sample))
        else:
            print(f"[WARN] Signal sample directory not found: {p}")

    for sample in BKG_SAMPLES:
        p = os.path.join(input_dir, "Bkg_MC", sample)
        if os.path.isdir(p):
            subdirs_to_scan.append(os.path.join("Bkg_MC", sample))
        else:
            print(f"[WARN] Bkg_MC sample directory not found: {p}")

    data_p = os.path.join(input_dir, "Data", "Data")
    if os.path.isdir(data_p):
        subdirs_to_scan.append(os.path.join("Data", "Data"))
    else:
        print(f"[WARN] Data directory not found: {data_p}")

    visited_dirs = set()

    def onerror(err):
        print(f"[WALK ERROR] {err}", file=sys.stderr)

    for relsub in subdirs_to_scan:
        search_root = os.path.join(input_dir, relsub)
        for dirpath, dirnames, filenames in os.walk(
            search_root, topdown=True, followlinks=True, onerror=onerror
        ):
            real_dir = os.path.realpath(dirpath)
            if real_dir in visited_dirs:
                dirnames[:] = []
                continue
            visited_dirs.add(real_dir)

            kept = []
            for dirname in dirnames:
                child = os.path.join(dirpath, dirname)
                if os.path.realpath(child) not in visited_dirs:
                    kept.append(dirname)
            dirnames[:] = kept

            for fname in filenames:
                if fname.lower().endswith(".root"):
                    files.append(os.path.abspath(os.path.join(dirpath, fname)))

    return sorted(set(files)), subdirs_to_scan


def merge_one_file(input_path, output_path, force=False, class_tag=None):
    """Merge one ROOT file and return a status dictionary.

    class_tag (int or None): the 4-class id to stamp on every event. Should be
    passed in by the caller (derived from the file path via class_tag_for()).
    If None, class_tag is NOT added (kept for backward-compatible calls).
    """
    result = {
        "status": None,
        "n_input": 0,
        "n_output": 0,
        "trees_found": 0,
        "trees_nonempty": 0,
        "message": "",
    }

    if os.path.exists(output_path) and not force:
        result["status"] = "EXISTS"
        result["message"] = "output already exists"
        return result

    if not os.path.isfile(input_path):
        result["status"] = "ERROR"
        result["message"] = "input path is not a regular file"
        return result

    if not os.access(input_path, os.R_OK):
        result["status"] = "UNREADABLE"
        result["message"] = "no read permission"
        return result

    parts = []

    try:
        with uproot.open(input_path) as f:
            available = set(f.keys(cycle=False))
            result["trees_found"] = sum(
                tree_name in available for _, tree_name in TREE_TAGS
            )

            if result["trees_found"] == 0:
                result["status"] = "NO_MATCHING_TREE"
                result["message"] = "none of the 10 expected trees exists"
                return result

            for tag, tree_name in TREE_TAGS:
                if tree_name not in available:
                    continue

                tree = f[tree_name]
                if tree.num_entries == 0:
                    continue

                result["trees_nonempty"] += 1

                if "weight_lumi" not in set(tree.keys()):
                    raise RuntimeError(f"{tree_name} has no weight_lumi branch")

                arr = tree.arrays(library="ak")
                n = len(arr)

                arr = ak.with_field(
                    arr,
                    ak.Array(np.full(n, tag, dtype=np.int32)),
                    "source_tree_tag",
                )

                w = ak.to_numpy(arr["weight_lumi"])
                w_abs = np.abs(np.asarray(w, dtype=np.float64)).astype(np.float32)
                arr = ak.with_field(arr, ak.Array(w_abs), "weight_lumi_abs")

                # photon_eta_fold = int32(photon_eta * 10000); fold variable.
                # Same encoding as the old add_photon_eta_fold.py, computed
                # here so no separate post-merge pass is needed.
                pe = ak.to_numpy(arr["photon_eta"])
                pe_fold = np.asarray(np.floor(pe * 10000.0), dtype=np.int32)
                arr = ak.with_field(arr, ak.Array(pe_fold), "photon_eta_fold")

                # class_tag: which 4-class this sample belongs to (same value
                # for every event in the file). Skipped only if the caller did
                # not pass one (kept for backward-compatible direct calls).
                if class_tag is not None:
                    arr = ak.with_field(
                        arr,
                        ak.Array(np.full(n, int(class_tag), dtype=np.int32)),
                        "class_tag",
                    )

                parts.append(arr)
                result["n_input"] += n

    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = f"read/merge error: {exc}"
        return result

    if not parts:
        result["status"] = "EMPTY_MATCHING_TREES"
        result["message"] = "matching trees exist but all are empty"
        return result

    try:
        merged = ak.concatenate(parts, axis=0)
    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = f"ak.concatenate failed: {exc}"
        return result

    n_total = len(merged)
    if n_total != result["n_input"]:
        result["status"] = "ERROR"
        result["message"] = (
            f"entry mismatch before write: input={result['n_input']} "
            f"merged={n_total}"
        )
        return result

    output_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = f"cannot create output directory: {exc}"
        return result

    if not os.access(output_dir, os.W_OK | os.X_OK):
        result["status"] = "ERROR"
        result["message"] = f"output directory not writable: {output_dir}"
        return result

    out_dict = {}
    for field in merged.fields:
        try:
            out_dict[field] = ak.to_numpy(merged[field])
        except Exception:
            out_dict[field] = merged[field]

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".merge_tmp_", suffix=".root", dir=output_dir
        )
        os.close(fd)
        os.unlink(tmp_path)

        with uproot.recreate(tmp_path) as f_out:
            f_out[OUTPUT_TREE] = out_dict

        with uproot.open(tmp_path) as f_check:
            if OUTPUT_TREE not in f_check:
                raise RuntimeError(f"{OUTPUT_TREE} missing after write")
            n_output = int(f_check[OUTPUT_TREE].num_entries)

        if n_output != n_total:
            raise RuntimeError(
                f"entry mismatch after write: expected={n_total}, "
                f"written={n_output}"
            )

        result["n_output"] = n_output
        os.replace(tmp_path, output_path)
        tmp_path = None

    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = f"write/verify error: {exc}"
        return result

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    result["status"] = "DONE"
    result["message"] = f"{n_total} events"
    return result


def write_manifest(path, rows):
    if not path:
        return

    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fout:
        fout.write(
            "status\tn_input\tn_output\ttrees_found\ttrees_nonempty\t"
            "input\toutput\tmessage\n"
        )
        for row in rows:
            fout.write(
                f"{row['status']}\t{row['n_input']}\t{row['n_output']}\t"
                f"{row['trees_found']}\t{row['trees_nonempty']}\t"
                f"{row['input']}\t{row['output']}\t{row['message']}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest",
        default="merge_trees_v3_status.txt",
        help="per-file status report; use '' to disable",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir.rstrip("/"))
    output_dir = os.path.abspath(args.output_dir.rstrip("/"))

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory does not exist: {input_dir}")
        sys.exit(1)

    if not os.access(input_dir, os.R_OK | os.X_OK):
        print(f"ERROR: input directory not readable/searchable: {input_dir}")
        sys.exit(1)

    files, subdirs_scanned = find_root_files(input_dir)
    if not files:
        print(f"ERROR: no ROOT files found under {input_dir}")
        sys.exit(1)

    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Subdirs scanned ({len(subdirs_scanned)}):")
    for s in subdirs_scanned:
        print(f"  {s}")
    print(f"Files found:  {len(files)}")
    print()

    if args.dry_run:
        for input_path in files:
            rel = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel)
            readable = os.access(input_path, os.R_OK)
            print(f"[{'READ' if readable else 'NO-READ'}] {input_path}")
            print(f"    -> {output_path}")
        print("\n[dry-run] nothing written.")
        return

    counters = {
        "DONE": 0,
        "EXISTS": 0,
        "UNREADABLE": 0,
        "NO_MATCHING_TREE": 0,
        "EMPTY_MATCHING_TREES": 0,
        "ERROR": 0,
    }
    manifest_rows = []
    t_start = time.time()

    for i, input_path in enumerate(files, 1):
        rel = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel)

        try:
            this_class_tag = class_tag_for(input_path, input_dir)
        except ValueError as exc:
            print(f"[{i}/{len(files)}] {input_path}")
            print(f"  [ERROR] {exc}")
            counters["ERROR"] = counters.get("ERROR", 0) + 1
            manifest_rows.append({
                "status": "ERROR", "n_input": 0, "n_output": 0,
                "trees_found": 0, "trees_nonempty": 0,
                "message": str(exc),
                "input": input_path, "output": output_path,
            })
            continue

        print(f"[{i}/{len(files)}] {input_path}  (class_tag={this_class_tag})")
        t0 = time.time()
        result = merge_one_file(
            input_path, output_path, force=args.force, class_tag=this_class_tag
        )
        elapsed = time.time() - t0

        status = result["status"]
        counters[status] = counters.get(status, 0) + 1

        if status == "DONE":
            print(f"  [DONE] {result['n_output']} events ({elapsed:.1f}s)")
        else:
            print(f"  [{status}] {result['message']} ({elapsed:.1f}s)")

        manifest_rows.append({**result, "input": input_path, "output": output_path})

    if args.manifest:
        write_manifest(args.manifest, manifest_rows)

    total_elapsed = time.time() - t_start

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Discovered             : {len(files)}")
    for status in [
        "DONE",
        "EXISTS",
        "UNREADABLE",
        "NO_MATCHING_TREE",
        "EMPTY_MATCHING_TREES",
        "ERROR",
    ]:
        print(f"{status:<22}: {counters.get(status, 0)}")
    print(f"Total time             : {total_elapsed:.0f}s")
    print(f"Output                 : {output_dir}")
    if args.manifest:
        print(f"Manifest               : {os.path.abspath(args.manifest)}")

    problems = counters.get("UNREADABLE", 0) + counters.get("ERROR", 0)
    if problems > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
