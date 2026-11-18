import sys
import os
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import root_gnn_base.utils as utils
from root_gnn_base.dataset import compute_cutflow, print_cutflow
import uproot
import awkward as ak
import argparse
from root_gnn_base.batched_dataset import PreBatchedDataset
from root_gnn_base.batched_dataset import LazyPreBatchedDataset


def print_dataset_cutflow(dset_config, dataset_name):
    args = dset_config["args"]
    raw_dir = args["raw_dir"]
    file_names = args["file_names"]
    tree_name = args.get("tree_name", "nominal_Loose")
    selections = dset_config.get("selections", [])

    files = []
    if isinstance(file_names, str):
        files = glob.glob(os.path.join(raw_dir, file_names))
    else:
        for file_name in file_names:
            files.extend(glob.glob(os.path.join(raw_dir, file_name)))

    if not files:
        raise FileNotFoundError(
            f"No files found for dataset {dataset_name} in "
            f"{os.path.join(raw_dir, str(file_names))}"
        )

    if not selections:
        total = 0
        for file in files:
            with uproot.open(file) as f:
                total += int(f[tree_name].num_entries)
        cutflow = {"selections": [], "counts": [], "total": total}
        print_cutflow(cutflow, title=f"Cutflow for {dataset_name}")
        return

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
    from root_gnn_base.dataset import selection_branches
    for selection in selections:
        branches.extend(selection_branches(selection))
    branches = sorted(set(branches))

    arrays = []
    for file in files:
        with uproot.open(file) as f:
            arrays.append(f[tree_name].arrays(branches, library="ak"))
    if not arrays:
        print(f"No files found for dataset {dataset_name} in {os.path.join(raw_dir, str(file_names))}")
        return

    data = ak.concatenate(arrays, axis=0)
    cutflow = compute_cutflow(data, selections)
    print_cutflow(cutflow, title=f"Cutflow for {dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', type=str, required=True)
    add_arg('--dataset', type=str, required=True)
    add_arg('--chunk', type=int, default=0)
    add_arg('--shuffle_mode', action='store_true', help='Shuffle the dataset before training.')
    add_arg('--drop_last', action='store_false', help='Set drop_last to False if the flag is provided. Defaults to True.')
    args = parser.parse_args()

    config = utils.load_config(args.config)
    dset_config = config['Datasets'][args.dataset]
    print_dataset_cutflow(dset_config, args.dataset)
    batch_size = config['Training']['batch_size']
    if not args.shuffle_mode:
        dset = utils.buildFromConfig(dset_config, {'process_chunks': [args.chunk,]})
    else:
        dset = utils.buildFromConfig(dset_config)
        if 'batch_size' in dset_config:
            batch_size = dset_config['batch_size']

        shuffle_chunks = dset_config.get('shuffle_chunks', 10)
        shuffle_seed = dset_config.get('shuffle_seed', 12345)
        padding_mode = dset_config.get('padding_mode', 'STEPS')
        fold_conf = dset_config["folding"]
        print(f"shuffle_chunks = {shuffle_chunks}, args.chunk = {args.chunk}, padding_mode = {padding_mode}, shuffle_seed = {shuffle_seed}")
        if dset_config["class"] == "LazyMultiLabelDataset":
            LazyPreBatchedDataset(start_dataset = dset, batch_size = batch_size, mask_fn = utils.fold_selection(fold_conf, "train"), suffix = utils.fold_selection_name(fold_conf, "train"), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last, hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed )
            LazyPreBatchedDataset(start_dataset = dset, batch_size = batch_size, mask_fn = utils.fold_selection(fold_conf, "test"),  suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last, hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed)
            if "validation" in fold_conf:
                LazyPreBatchedDataset(start_dataset = dset, batch_size = batch_size, mask_fn = utils.fold_selection(fold_conf, "validation"), suffix = utils.fold_selection_name(fold_conf, 'validation'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last, hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed)

        else:
            PreBatchedDataset(dset, batch_size, utils.fold_selection(fold_conf, "train"), suffix = utils.fold_selection_name(fold_conf, "train"), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last,hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed)
            PreBatchedDataset(dset, batch_size, utils.fold_selection(fold_conf, "test"),  suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last,hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed )
            if "validation" in fold_conf:
                PreBatchedDataset(dset, batch_size, utils.fold_selection(fold_conf, "validation"), suffix = utils.fold_selection_name(fold_conf, 'validation'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode, drop_last=args.drop_last, hidden_size=config['Model']['args']['hid_size'], shuffle_seed=shuffle_seed)

if __name__ == "__main__":
    main()
