#!/usr/bin/env python3
"""
prepare_pt_only_inputs_v3.py

Second step of the v3 4-class data preparation: read merged_SR_Sideband trees
from root_p2root_v3_merged and write slim pt-only inputs with derived pT/m_llg
branches plus the per-process average weight, to root_p2root_v3_pt_only.

What this does
--------------
For every merged file it:

  1. Adds a derived branch `<pt_branch>_div_mllg = pT / m_llg` for each of the
     training pT branches (lep0-5, photon, jet0-5, bjet0-1, MET). The config
     later chooses which of these (and which raw pT) actually feed the GNN, so
     this script keeps both options available. This mirrors the v2
     prepare_pt_only_inputs.py behaviour.

  2. Adds branch `weight_lumi_avg` (float32): a single constant per physics
     process, equal to mean(weight_lumi) over all era files in that process
     directory. This replaces the separate add_avg_weight_and_check.py step
     from v2. The constant is the same for every event of that process, which
     makes weighted yields equal to N_effective-like counts and stabilises
     training against large per-event weight fluctuations (notably in
     DYJetsToLL_Stitch). It is also the weight variable the significance / Z
     scripts consume.

  3. Carries over every original branch unchanged (source_tree_tag,
     photon_eta_fold, weight_lumi, weight_lumi_abs, m_llg, ...).

Process grouping for weight_lumi_avg
------------------------------------
The mean is computed per sample DIRECTORY (Signal/ttH, Signal/VH, ...,
Bkg_MC/DYGto2LG_600, Bkg_MC/DYJetsToLL_Stitch) across ALL era .root files that
exist under it. v3 includes Run2+Run3 (2016preVFP .. 2026); not every sample
has every era, and this script only averages over the eras physically present.
Data files: in v3 the Data trees DO carry a weight_lumi branch (constant 1.0
per event), so they are treated like any MC sample -- weight_lumi_avg is
computed and written (it will be 1.0). Data is still never used for training;
the weight is just there so the schema is uniform. The None/NaN fallback below
only triggers if some future Data file genuinely lacks weight_lumi.

Inputs / outputs
----------------
--input-dir   default /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_merged
--output-dir  default /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only

The output directory must not already exist (refuse to overwrite), matching
the safety behaviour of the v2 script.

Usage:
    python prepare_pt_only_inputs_v3.py
    python prepare_pt_only_inputs_v3.py --input-dir /path --output-dir /path
    python prepare_pt_only_inputs_v3.py --step-size 50000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import uproot

DEFAULT_TREE = "merged_SR_Sideband"

# Training pT branches: leptons (0-5), photon, jets (0-5), bjets (0-1), MET.
# Same set as v2 prepare_pt_only_inputs.py. jet6-9 exist in v3 but are not
# used by the current GNN node layout (6 jets max), so they are not divided.
PT_BRANCHES = tuple(
    [f"lep{i}_pt" for i in range(6)]
    + ["photon_pt"]
    + [f"jet{i}_pt" for i in range(6)]
    + [f"bjet{i}_pt" for i in range(2)]
    + ["MET_pt"]
)


def find_tree(root_file, preferred):
    """Locate the merged tree, tolerant of ;1 cycle suffixes."""
    classnames = root_file.classnames(recursive=True)
    trees = [name for name, cls in classnames.items() if cls.startswith("TTree")]
    matches = [name for name in trees if name.rstrip(";1") == preferred]
    if matches:
        return matches[0]
    if len(trees) == 1:
        return trees[0]
    raise ValueError(f"Could not select a unique TTree; found {trees}. Pass --tree.")


def derived_arrays(arrays):
    """Compute <pt>_div_mllg for every PT_BRANCHES entry."""
    mass = np.asarray(arrays["m_llg"])
    if not np.all(np.isfinite(mass) & (mass > 0)):
        raise ValueError("m_llg contains nonfinite or nonpositive values")
    out = {}
    for branch in PT_BRANCHES:
        pt = np.asarray(arrays[branch])
        if not np.all(np.isfinite(pt)):
            raise ValueError(f"{branch} contains nonfinite values")
        out[f"{branch}_div_mllg"] = np.where(pt > 0, pt / mass, pt).astype(np.float32)
    return out


def compute_avg_weights(input_dir):
    """Pass 1: scan every file, compute mean(weight_lumi) per sample dir.

    Returns {rel_path_of_sample_dir: avg_weight_float}. Data files are
    detected (no weight_lumi branch) and mapped to None so we still write a
    placeholder column for them.

    rel_path here is the directory containing the .root file, relative to
    input_dir, e.g. 'Signal/ttH', 'Bkg_MC/DYJetsToLL_Stitch', 'Data/Data'.
    """
    files = sorted(input_dir.rglob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files under {input_dir}")

    # group files by their immediate parent dir (the sample dir)
    groups = {}
    for src in files:
        rel_dir = src.parent.relative_to(input_dir).as_posix()
        groups.setdefault(rel_dir, []).append(src)

    avg_weights = {}
    for rel_dir, group_files in sorted(groups.items()):
        total_w = 0.0
        total_n = 0
        is_data = False
        for f in group_files:
            with uproot.open(f) as root_file:
                tree_path = find_tree(root_file, DEFAULT_TREE)
                tree = root_file[tree_path]
                branches = set(tree.keys())
                if "weight_lumi" not in branches:
                    # Data: no per-event weight. Mark and skip; we still need
                    # to write a placeholder column later.
                    is_data = True
                    continue
                # iterate to keep memory bounded on the large Stitch files
                for arr in tree.iterate(
                    ["weight_lumi"], step_size=50000, library="np"
                ):
                    w = np.asarray(arr["weight_lumi"], dtype=np.float64)
                    total_w += float(np.sum(w))
                    total_n += int(w.size)
        if is_data and total_n == 0:
            avg_weights[rel_dir] = None
            print(f"  [avg] {rel_dir:30s} DATA -> weight_lumi_avg = NaN placeholder")
        else:
            avg = (total_w / total_n) if total_n > 0 else 0.0
            avg_weights[rel_dir] = avg
            print(
                f"  [avg] {rel_dir:30s} N={total_n:8d} sum(w)={total_w:.6f} "
                f"avg={avg:.8f}"
            )
    return avg_weights


def process_file(source, destination, preferred_tree, step_size, avg_weight):
    """Copy one merged file, adding _div_mllg branches and weight_lumi_avg.

    avg_weight is the per-sample-dir constant (already computed in pass 1) or
    None for Data files (weight_lumi_avg will be a NaN placeholder).
    """
    with uproot.open(source) as root_file:
        tree_path = find_tree(root_file, preferred_tree)
        tree = root_file[tree_path]
        branches = set(tree.keys())

        missing = sorted({"m_llg", *PT_BRANCHES} - branches)
        if missing:
            raise ValueError(f"{source}: missing required branches {missing}")

        new_pt_names = {f"{b}_div_mllg" for b in PT_BRANCHES}
        collision = sorted(new_pt_names & branches)
        if collision:
            raise ValueError(
                f"{source}: derived branches already exist: {collision}"
            )

        has_wlumi = "weight_lumi" in branches
        if avg_weight is None and has_wlumi:
            # Should not happen for Data, but guard against surprises.
            raise ValueError(
                f"{source}: avg_weight is None but file has weight_lumi"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            temporary.unlink()

        # placeholder value for Data (no weight): NaN, written as float32
        avg_val = float("nan") if avg_weight is None else float(avg_weight)

        with uproot.recreate(temporary) as output:
            first = tree.arrays(entry_stop=0, library="np")
            schema = {name: value.dtype for name, value in first.items()}
            schema.update({name: np.dtype("float32") for name in new_pt_names})
            schema["weight_lumi_avg"] = np.dtype("float32")
            out_tree = output.mktree(
                tree_path.rstrip(";1"), schema, title=tree.title
            )

            count = 0
            requested = list(branches)
            for arrays in tree.iterate(requested, step_size=step_size, library="np"):
                arrays.update(derived_arrays(arrays))
                n = len(arrays["m_llg"])
                arrays["weight_lumi_avg"] = np.full(
                    n, avg_val, dtype=np.float32
                )
                out_tree.extend(arrays)
                count += n
            if count != tree.num_entries:
                raise RuntimeError(
                    f"{source}: copied {count} of {tree.num_entries} entries"
                )

        # verify the written file before rename
        with uproot.open(temporary) as check:
            check_tree = check[tree_path.rstrip(";1")]
            if check_tree.num_entries != count:
                raise RuntimeError(
                    f"{source}: verify failed, {check_tree.num_entries} != {count}"
                )

        temporary.replace(destination)
        return count, avg_val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_merged"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v3_pt_only"
        ),
    )
    parser.add_argument("--tree", default=DEFAULT_TREE)
    parser.add_argument("--step-size", type=int, default=20000)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output_dir}"
        )

    files = sorted(args.input_dir.rglob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files under {args.input_dir}")

    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Files:      {len(files)}")
    print()
    print("=== Pass 1: compute weight_lumi_avg per sample dir ===")
    avg_weights = compute_avg_weights(args.input_dir)
    print()

    print("=== Pass 2: write pt_only files with _div_mllg + weight_lumi_avg ===")
    records = []
    for index, source in enumerate(files, 1):
        rel_dir = source.parent.relative_to(args.input_dir).as_posix()
        avg_w = avg_weights.get(rel_dir)
        destination = args.output_dir / source.relative_to(args.input_dir)
        entries, avg_val = process_file(
            source, destination, args.tree, args.step_size, avg_w
        )
        avg_str = "NaN(Data)" if math.isnan(avg_val) else f"{avg_val:.8f}"
        records.append(
            {
                "file": str(source.relative_to(args.input_dir)),
                "entries": entries,
                "weight_lumi_avg": avg_str,
            }
        )
        print(
            f"{index}/{len(files)} {source.relative_to(args.input_dir)}: "
            f"{entries} events, w_avg={avg_str}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "tree": args.tree,
        "derived_branches": [f"{b}_div_mllg" for b in PT_BRANCHES],
        "added_branches": ["weight_lumi_avg"],
        "weight_lumi_avg_by_sample_dir": {
            k: (None if v is None else v) for k, v in avg_weights.items()
        },
        "files": records,
        "total_entries": sum(r["entries"] for r in records),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print()
    print(
        f"COMPLETE {len(files)} files, {manifest['total_entries']} events"
    )
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
