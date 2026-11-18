#!/usr/bin/env python3
"""
merge_cr_lowIDMVA_v3.py

Merge the CR_lowIDMVA control-region trees from v3 Data into a single
inference-ready tree per era, mirroring merge_trees_v3.py but restricted to
the Data sample and the CR_lowIDMVA region. This is the data-prep step for
the method-2 (control-region) background estimate.

What this does
--------------
For every era .root file under Data/Data of the v3 Mll80to100 sample, this
merges the five analysis categories' CR_lowIDMVA SR + CR_lowIDMVA_SB trees
into one TTree called "merged_CR_lowIDMVA".

Method-2 background estimate needs BOTH region flavours per category:
  - CR_lowIDMVA__<cat>     (CR SR region, m_llg ~ 120-130): the mass shape and
                           the 120-130 yield after scaling (step 3).
  - CR_lowIDMVA_SB__<cat>  (CR SB region, m_llg ~ 95-180): the sideband
                           integral used to scale CR data onto the nominal
                           sideband data (step 2).

source_tree_tag (int32) records which input tree an event came from. Following
the user's chosen layout, the five CR-SR categories come first (0-4) and the
five CR-SB categories after (5-9):
    0  CR_lowIDMVA__ttHhad        (CR SR)
    1  CR_lowIDMVA__ttHlep
    2  CR_lowIDMVA__untag
    3  CR_lowIDMVA__VH3l
    4  CR_lowIDMVA__VHpTmiss
    5  CR_lowIDMVA_SB__ttHhad     (CR SB)
    6  CR_lowIDMVA_SB__ttHlep
    7  CR_lowIDMVA_SB__untag
    8  CR_lowIDMVA_SB__VH3l
    9  CR_lowIDMVA_SB__VHpTmiss

Note this differs from merge_trees_v3.py, where SR/SB alternate (2i / 2i+1).
Here SR and SB are grouped contiguously because method 2 treats the two
region sets as whole populations (a shape vs. an integral), not as paired
per-category regions. Downstream scripts must read source_tree_tag with this
layout in mind.

Other added columns (identical semantics to merge_trees_v3.py):
    weight_lumi_abs  (float32) -- abs(weight_lumi); v3 Data carries 1.0
    photon_eta_fold  (int32)   -- int32(photon_eta * 10000); the fold variable
                                  (used by inference to split events by fold,
                                  same encoding as nominal so the same
                                  photon_eta_fold % n_folds logic applies).
    class_tag        (int32)   -- -1 (Data is never a training class)

The inclusive trees CR_lowIDMVA__inclusive / CR_lowIDMVA_SB__inclusive are NOT
merged. The five category trees are exclusive slices; the inclusive tree
double-counts them (verified on 2018: inclusive=10009 vs sum(5cat)=161 for
CR SR). The inclusive tree stays in the raw file for cross-checks only.

Only Data/Data is scanned. Method 2 is fully data-driven; MC is not
reprocessed here (nominal MC and nominal Data pipelines are already complete
and are left untouched).

Branch schema: CR trees share the exact same 406 branches as SR/Sideband
trees (verified), so the downstream prepare_pt_only_inputs_v3.py can be
reused unchanged with --tree merged_CR_lowIDMVA.

Usage:
    python merge_cr_lowIDMVA_v3.py
    python merge_cr_lowIDMVA_v3.py --force
    python merge_cr_lowIDMVA_v3.py --dry-run
    python merge_cr_lowIDMVA_v3.py --input-dir /path/to/root_p2root_v3_Mll80to100
                                  --output-dir /path/to/root_p2root_v3_cr_merged
    python merge_cr_lowIDMVA_v3.py --manifest merge_cr_lowIDMVA_status.txt
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
DEFAULT_OUTPUT_DIR = os.path.join(REPO, "root_p2root_v3_cr_merged")
OUTPUT_TREE = "merged_CR_lowIDMVA"

# Five analysis categories, identical to the nominal setup.
REGIONS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]

# source_tree_tag layout: CR SR categories 0-4, CR SB categories 5-9 (grouped,
# not interleaved -- see module docstring for the rationale).
TREE_TAGS = []
for _i, _r in enumerate(REGIONS):
    TREE_TAGS.append((_i, f"CR_lowIDMVA__{_r}"))
for _i, _r in enumerate(REGIONS):
    TREE_TAGS.append((5 + _i, f"CR_lowIDMVA_SB__{_r}"))

# Data only. class_tag is -1 for every event (Data is never trained).
DATA_CLASS_TAG = -1


def find_root_files(input_dir):
    """Find all .root files under Data/Data.

    Returns sorted list of absolute paths. Mirrors the walk logic of
    merge_trees_v3.find_root_files but scoped to the single Data directory.
    """
    data_p = os.path.join(input_dir, "Data", "Data")
    if not os.path.isdir(data_p):
        print(f"[ERROR] Data directory not found: {data_p}")
        return []

    files = []
    visited_dirs = set()

    def onerror(err):
        print(f"[WALK ERROR] {err}", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(
        data_p, topdown=True, followlinks=True, onerror=onerror
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

    return sorted(set(files))


def merge_one_file(input_path, output_path, force=False, class_tag=DATA_CLASS_TAG):
    """Merge one Data ROOT file's CR_lowIDMVA trees and return a status dict.

    Identical I/O discipline to merge_trees_v3.merge_one_file: write to a temp
    file in the output dir, reopen to verify entry count, then atomically rename.
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
                result["message"] = "none of the 10 expected CR trees exists"
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
                # Same encoding as nominal so inference can split CR Data by
                # fold with the identical photon_eta_fold % n_folds logic.
                pe = ak.to_numpy(arr["photon_eta"])
                pe_fold = np.asarray(np.floor(pe * 10000.0), dtype=np.int32)
                arr = ak.with_field(arr, ak.Array(pe_fold), "photon_eta_fold")

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
        default="merge_cr_lowIDMVA_status.txt",
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

    files = find_root_files(input_dir)
    if not files:
        print(f"ERROR: no ROOT files found under {input_dir}/Data/Data")
        sys.exit(1)

    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Output tree:  {OUTPUT_TREE}")
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

        print(f"[{i}/{len(files)}] {input_path}  (class_tag={DATA_CLASS_TAG})")
        t0 = time.time()
        result = merge_one_file(
            input_path, output_path, force=args.force, class_tag=DATA_CLASS_TAG
        )
        elapsed = time.time() - t0

        status = result["status"]
        counters[status] = counters.get(status, 0) + 1

        if status == "DONE":
            print(
                f"  [DONE] {result['n_output']} events "
                f"(trees_nonempty={result['trees_nonempty']}/10) ({elapsed:.1f}s)"
            )
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
