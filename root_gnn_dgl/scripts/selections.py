import argparse
import glob
import os
import sys

import awkward as ak
import uproot

file_path = os.getcwd()
sys.path.append(file_path)

from root_gnn_base import utils
from root_gnn_base.dataset import selection_branches, check_selection, print_cutflow, init_cutflow


def get_branches(dataset_config):
    args = dataset_config["args"]
    branches = []
    for feat in args.get("node_branch_names", []):
        if isinstance(feat, list):
            for branch in feat:
                if isinstance(branch, str) and branch != "CALC_E":
                    branches.append(branch)
    for feat in args.get("global_features", []):
        if isinstance(feat, str):
            branches.append(feat)
    for feat in args.get("tracking_info", []):
        if isinstance(feat, str):
            branches.append(feat)
    for selection in dataset_config.get("selections", []):
        branches.extend(selection_branches(selection))
    return sorted(set(branches))


def load_arrays(dataset_config):
    args = dataset_config["args"]
    raw_dir = args["raw_dir"]
    file_names = args["file_names"]
    tree_name = args.get("tree_name", "nominal_Loose")

    files = []
    if isinstance(file_names, str):
        files = glob.glob(os.path.join(raw_dir, file_names))
    else:
        for file_name in file_names:
            files.extend(glob.glob(os.path.join(raw_dir, file_name)))

    branches = get_branches(dataset_config)
    arrays = []
    for file in files:
        with uproot.open(file) as f:
            arrays.append(f[tree_name].arrays(branches, library="ak"))
    if not arrays:
        raise FileNotFoundError(f"No files found in {os.path.join(raw_dir, str(file_names))}")
    return ak.concatenate(arrays, axis=0), branches


def vectorized_cutflow(data, selections):
    cutflow = init_cutflow(selections)
    first_field = data.fields[0] if len(data.fields) > 0 else None
    cutflow["total"] = len(data[first_field]) if first_field is not None else 0
    mask = ak.ones_like(data[first_field], dtype=bool) if first_field is not None else None

    for i, selection in enumerate(selections):
        if isinstance(selection, str):
            current_mask = eval(selection, {"__builtins__": {}}, data)
        else:
            current_mask = check_selection(data, selection)
        current_mask = ak.to_numpy(current_mask)
        if mask is None:
            mask = current_mask
        else:
            mask = mask & current_mask
        cutflow["counts"][i] = int(ak.sum(mask))
    return cutflow


def main():
    parser = argparse.ArgumentParser(description="Fast selection tester and cutflow printer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    for dataset_name, dataset_config in config["Datasets"].items():
        selections = dataset_config.get("selections", [])
        print(f"\n== Dataset: {dataset_name} ==")

        data, branches = load_arrays(dataset_config)

        for selection in selections:
            try:
                _ = eval(selection, {"__builtins__": {}}, data) if isinstance(selection, str) else check_selection({b: data[b] for b in branches}, selection)
                print(f"OK: {selection}")
            except Exception:
                print(f"FAILED: {selection}")
                raise

        cutflow = vectorized_cutflow(data, selections)
        print_cutflow(cutflow, title=f"Cutflow for {dataset_name}")


if __name__ == "__main__":
    main()
