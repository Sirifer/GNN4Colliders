#!/usr/bin/env python3
"""
Combine the three CR inclusive trees into a single merged_CR tree per era.

  cr_type_tag (int32): 0=CR_lowIDMVA  1=CR_lowPTOverMllg  2=CR_FSRgamma

Usage:
    python merge_cr_trees.py
    python merge_cr_trees.py --input-dir /path/to/Data --output-dir /path/to/out
    python merge_cr_trees.py --force
"""

import argparse
import os
import numpy as np
import awkward as ak
import uproot

REPO = "/scratchfs2/cms/huangsuyun/GNN4Colliders"
DEFAULT_INPUT_DIR  = os.path.join(REPO, "root_p2root_v2/Data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO, "root_p2root_v2_cr_merged")

CR_TYPES = [
    (0, "CR_lowIDMVA__inclusive"),
    (1, "CR_lowPTOverMllg__inclusive"),
    (2, "CR_FSRgamma__inclusive"),
]
OUT_TREE = "merged_CR"
ERAS     = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]


def merge_one(src, dst, force=False):
    if os.path.exists(dst) and not force:
        print("  [SKIP] already exists: %s" % dst)
        return 0

    parts = []
    with uproot.open(src) as f:
        available = set(f.keys(cycle=False))
        for cr_idx, tree_name in CR_TYPES:
            if tree_name not in available:
                print("  [WARN] missing %s" % tree_name)
                continue
            t = f[tree_name]
            if t.num_entries == 0:
                continue
            arr = t.arrays(library="ak")
            n = len(arr)
            arr = ak.with_field(arr, ak.Array(np.full(n, cr_idx, dtype=np.int32)), "cr_type_tag")
            parts.append(arr)

    if not parts:
        print("  [WARN] no non-empty CR trees, skipping")
        return 0

    merged = ak.concatenate(parts, axis=0)
    out_dict = {}
    for field in merged.fields:
        try:
            out_dict[field] = ak.to_numpy(merged[field])
        except Exception:
            out_dict[field] = merged[field]

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".partial"
    with uproot.recreate(tmp) as fout:
        fout[OUT_TREE] = out_dict
    os.replace(tmp, dst)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("cr_type_tag: 0=CR_lowIDMVA  1=CR_lowPTOverMllg  2=CR_FSRgamma")
    print()

    for era in ERAS:
        src = os.path.join(args.input_dir,  "%s.root" % era)
        dst = os.path.join(args.output_dir, "%s.root" % era)
        print("[%s]" % era)
        if not os.path.exists(src):
            print("  [SKIP] missing source: %s" % src)
            continue
        n = merge_one(src, dst, force=args.force)
        if n:
            print("  [DONE] %d entries -> %s" % (n, dst))

    print("\nDone.")


if __name__ == "__main__":
    main()
