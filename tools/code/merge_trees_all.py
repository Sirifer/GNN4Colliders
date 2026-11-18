# """
# merge_trees_all.py

# Recursively finds all ROOT files under root_p2root_v2/Signal and
# root_p2root_v2/Bkg_MC, merges the 5 non-ggH/VBF regions (SR + Sideband)
# into a single TTree called "merged_SR_Sideband", and writes the output
# to root_p2root_v2_merged/ preserving the same directory structure.

# Two extra columns are added to the merged tree:
#   source_tree_tag  (int32)  -- which input tree the event came from
#                                0  SR__ttHhad       1  Sideband__ttHhad
#                                2  SR__ttHlep       3  Sideband__ttHlep
#                                4  SR__untag        5  Sideband__untag
#                                6  SR__VH3l         7  Sideband__VH3l
#                                8  SR__VHpTmiss     9  Sideband__VHpTmiss
#   weight_lumi_abs  (float32) -- abs(weight_lumi), for use as training weight

# Usage:
#     python merge_trees_all.py                    # process all files
#     python merge_trees_all.py --force            # overwrite existing outputs
#     python merge_trees_all.py --dry-run          # print file list, do nothing
#     python merge_trees_all.py --input-dir /path/to/root_p2root_v2
#                               --output-dir /path/to/root_p2root_v2_merged
# """

# import argparse
# import os
# import sys
# import time

# import awkward as ak
# import numpy as np
# import uproot

# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# REPO = "/scratchfs2/cms/huangsuyun/GNN4Colliders"
# DEFAULT_INPUT_DIR  = os.path.join(REPO, "root_p2root_v2")
# DEFAULT_OUTPUT_DIR = os.path.join(REPO, "root_p2root_v2_merged")

# REGIONS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]

# TREE_TAGS = []
# for _i, _r in enumerate(REGIONS):
#     TREE_TAGS.append((2 * _i,     "SR__%s"       % _r))
#     TREE_TAGS.append((2 * _i + 1, "Sideband__%s" % _r))

# # ---------------------------------------------------------------------------
# # Core merge logic
# # ---------------------------------------------------------------------------

# def merge_one_file(input_path, output_path, force=False):
#     """Read all region trees from input_path, merge, write to output_path."""

#     if os.path.exists(output_path) and not force:
#         print("  [SKIP] already exists: %s" % output_path)
#         return 0

#     parts = []
#     with uproot.open(input_path) as f:
#         available = set(f.keys(cycle=False))
#         for tag, tree_name in TREE_TAGS:
#             if tree_name not in available:
#                 continue
#             t = f[tree_name]
#             if t.num_entries == 0:
#                 continue
#             arr = t.arrays(library="ak")
#             n = len(arr)
#             arr["source_tree_tag"] = ak.Array(np.full(n, tag, dtype=np.int32))
#             arr["weight_lumi_abs"] = ak.Array(
#                 np.abs(ak.to_numpy(arr["weight_lumi"]).astype(np.float32))
#             )
#             parts.append(arr)

#     if not parts:
#         print("  [WARN] no matching trees in %s — skipping" % input_path)
#         return 0

#     merged = ak.concatenate(parts, axis=0)
#     n_total = len(merged)

#     out_dict = {}
#     for field in merged.fields:
#         try:
#             out_dict[field] = ak.to_numpy(merged[field])
#         except Exception:
#             out_dict[field] = merged[field]

#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
#     with uproot.recreate(output_path) as f_out:
#         f_out["merged_SR_Sideband"] = out_dict

#     return n_total

# # ---------------------------------------------------------------------------
# # File discovery
# # ---------------------------------------------------------------------------

# def find_root_files(input_dir):
#     """Return sorted list of all .root files under Signal/ and Bkg_MC/."""
#     files = []
#     for subdir in ["Signal", "Bkg_MC"]:
#         search_root = os.path.join(input_dir, subdir)
#         if not os.path.isdir(search_root):
#             print("[WARN] directory not found: %s" % search_root)
#             continue
#         for dirpath, _, filenames in os.walk(search_root):
#             for fname in filenames:
#                 if fname.endswith(".root"):
#                     files.append(os.path.join(dirpath, fname))
#     return sorted(files)

# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR,
#                         help="Root of the original data (default: %s)" % DEFAULT_INPUT_DIR)
#     parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
#                         help="Root of the merged output (default: %s)" % DEFAULT_OUTPUT_DIR)
#     parser.add_argument("--force",   action="store_true",
#                         help="Overwrite output files that already exist")
#     parser.add_argument("--dry-run", action="store_true",
#                         help="Print what would be done but do nothing")
#     args = parser.parse_args()

#     input_dir  = args.input_dir.rstrip("/")
#     output_dir = args.output_dir.rstrip("/")

#     files = find_root_files(input_dir)
#     if not files:
#         print("ERROR: no ROOT files found under %s" % input_dir)
#         sys.exit(1)

#     print("Input dir:  %s" % input_dir)
#     print("Output dir: %s" % output_dir)
#     print("Files found: %d" % len(files))
#     print()

#     if args.dry_run:
#         for f in files:
#             out = output_dir + f[len(input_dir):]
#             print("  %s\n    -> %s" % (f, out))
#         print("\n[dry-run] nothing written.")
#         return

#     t_start = time.time()
#     n_done = 0
#     n_skip = 0
#     n_fail = 0

#     for i, input_path in enumerate(files, 1):
#         # derive output path by replacing input_dir prefix with output_dir
#         rel = input_path[len(input_dir):]
#         output_path = output_dir + rel

#         print("[%d/%d] %s" % (i, len(files), input_path))
#         try:
#             t0 = time.time()
#             n = merge_one_file(input_path, output_path, force=args.force)
#             elapsed = time.time() - t0
#             if n > 0:
#                 print("  -> %d events  (%.1fs)" % (n, elapsed))
#                 n_done += 1
#             else:
#                 n_skip += 1
#         except Exception as e:
#             print("  [ERROR] %s" % e)
#             n_fail += 1

#     total_elapsed = time.time() - t_start
#     print()
#     print("=" * 50)
#     print("Done.  processed=%d  skipped=%d  errors=%d  total_time=%.0fs"
#           % (n_done, n_skip, n_fail, total_elapsed))
#     print("Output: %s" % output_dir)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
merge_trees_all_robust.py

Recursively find ROOT files under <input-dir>/Signal and <input-dir>/Bkg_MC,
merge the five SR + Sideband region pairs into merged_SR_Sideband, and preserve
input directory structure in the output directory.

Improvements over the original script:
  * follows symlink directories and avoids simple symlink loops
  * reports os.walk / permission problems
  * distinguishes DONE / EXISTS / UNREADABLE / NO_MATCHING_TREE /
    EMPTY_MATCHING_TREES / ERROR
  * checks weight_lumi before creating weight_lumi_abs
  * uses os.path.relpath instead of string slicing for output paths
  * writes to a temporary ROOT file, verifies entry count, then renames it
  * writes a per-file manifest/status report
  * matches .root case-insensitively

Usage:
  python merge_trees_all_robust.py
  python merge_trees_all_robust.py --force
  python merge_trees_all_robust.py --dry-run
  python merge_trees_all_robust.py --manifest merge_status.txt
"""

import argparse
import os
import sys
import tempfile
import time

import awkward as ak
import numpy as np
import uproot
BASE = "/publicfs/cms/user/laipeizhu/Eos/HZgamma/"
REPO = "/scratchfs2/cms/huangsuyun/GNN4Colliders"
DEFAULT_INPUT_DIR = os.path.join(BASE, "root_p2root_v2/Data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO, "root_p2root_v2_merged_data")
OUTPUT_TREE = "merged_SR_Sideband"

REGIONS = ["ttHhad", "ttHlep", "untag", "VH3l", "VHpTmiss"]
TREE_TAGS = []
for i, region in enumerate(REGIONS):
    TREE_TAGS.append((2 * i, f"SR__{region}"))
    TREE_TAGS.append((2 * i + 1, f"Sideband__{region}"))


def find_root_files(input_dir):
    """Recursively find ROOT files under Signal/ and Bkg_MC/."""
    files = []
    visited_dirs = set()

    def onerror(err):
        print(f"[WALK ERROR] {err}", file=sys.stderr)

    for subdir in ["Data"]:
        search_root = os.path.join(input_dir, subdir)
        if not os.path.isdir(search_root):
            print(f"[WARN] directory not found: {search_root}")
            continue

        for dirpath, dirnames, filenames in os.walk(
            search_root,
            topdown=True,
            followlinks=True,
            onerror=onerror,
        ):
            real_dir = os.path.realpath(dirpath)
            if real_dir in visited_dirs:
                dirnames[:] = []
                continue
            visited_dirs.add(real_dir)

            # prune obvious symlink loops / repeated physical directories
            kept = []
            for dirname in dirnames:
                child = os.path.join(dirpath, dirname)
                if os.path.realpath(child) not in visited_dirs:
                    kept.append(dirname)
            dirnames[:] = kept

            for fname in filenames:
                if fname.lower().endswith(".root"):
                    files.append(os.path.abspath(os.path.join(dirpath, fname)))

    # Deduplicate only identical logical paths.
    return sorted(set(files))


def merge_one_file(input_path, output_path, force=False):
    """Merge one ROOT file and return a status dictionary."""
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
            f"entry mismatch before write: input={result['n_input']} merged={n_total}"
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
                f"entry mismatch after write: expected={n_total}, written={n_output}"
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
        default="merge_trees_all_status.txt",
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
        print(f"ERROR: no ROOT files found under {input_dir}")
        sys.exit(1)

    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Files found: {len(files)}")
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

        print(f"[{i}/{len(files)}] {input_path}")
        t0 = time.time()
        result = merge_one_file(input_path, output_path, force=args.force)
        elapsed = time.time() - t0

        status = result["status"]
        counters[status] = counters.get(status, 0) + 1

        if status == "DONE":
            print(f"  [DONE] {result['n_output']} events ({elapsed:.1f}s)")
        else:
            print(f"  [{status}] {result['message']} ({elapsed:.1f}s)")

        manifest_rows.append({
            **result,
            "input": input_path,
            "output": output_path,
        })

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