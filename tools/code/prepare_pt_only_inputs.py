import argparse
import json
from pathlib import Path
import numpy as np
import uproot

DEFAULT_TREE = "merged_SR_Sideband"
PT_BRANCHES = tuple(
    [f"lep{i}_pt" for i in range(6)]
    + ["photon_pt"]
    + [f"jet{i}_pt" for i in range(6)]
+ [f"bjet{i}_pt" for i in range(2)]
    + ["MET_pt"]
)

def find_tree(root_file, preferred):
    classnames = root_file.classnames(recursive=True)
    trees = [name for name, cls in classnames.items() if cls.startswith("TTree")]
    matches = [name for name in trees if name.rstrip(";1") == preferred]
    if matches:
        return matches[0]
    if len(trees) == 1:
        return trees[0]
    raise ValueError(f"Could not select a unique TTree; found {trees}. Pass --tree.")

def derived_arrays(arrays):
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

def process_file(source, destination, preferred_tree, step_size):
    with uproot.open(source) as root_file:
        tree_path = find_tree(root_file, preferred_tree)
        tree = root_file[tree_path]
        branches = set(tree.keys())
        missing = sorted({"m_llg", *PT_BRANCHES} - branches)
        if missing:
            raise ValueError(f"{source}: missing required branches {missing}")
        new_names = {f"{branch}_div_mllg" for branch in PT_BRANCHES}
        collision = sorted(new_names & branches)
        if collision:
            raise ValueError(f"{source}: derived branches already exist: {collision}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            temporary.unlink()
        with uproot.recreate(temporary) as output:
            first = tree.arrays(entry_stop=0, library="np")
            schema = {name: value.dtype for name, value in first.items()}
            schema.update({name: np.dtype("float32") for name in new_names})
            out_tree = output.mktree(tree_path.rstrip(";1"), schema, title=tree.title)
            count = 0
            requested = list(branches)
            for arrays in tree.iterate(requested, step_size=step_size, library="np"):
                arrays.update(derived_arrays(arrays))
                out_tree.extend(arrays)
                count += len(arrays["m_llg"])
            if count != tree.num_entries:
                raise RuntimeError(f"{source}: copied {count} of {tree.num_entries} entries")
        temporary.replace(destination)
        return count

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree", default=DEFAULT_TREE)
    parser.add_argument("--step-size", type=int, default=20000)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    files = sorted(args.input_dir.rglob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files under {args.input_dir}")
    records = []
    for index, source in enumerate(files, 1):
        destination = args.output_dir / source.relative_to(args.input_dir)
        entries = process_file(source, destination, args.tree, args.step_size)
        records.append({"file": str(source.relative_to(args.input_dir)), "entries": entries})
        print(f"{index}/{len(files)} {source.relative_to(args.input_dir)}: {entries}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"input_dir": str(args.input_dir.resolve()), "tree": args.tree,
                "derived_branches": [f"{branch}_div_mllg" for branch in PT_BRANCHES],
                "files": records, "total_entries": sum(r["entries"] for r in records)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETE {len(files)} files, {manifest['total_entries']} events", flush=True)

if __name__ == "__main__":
    main()
