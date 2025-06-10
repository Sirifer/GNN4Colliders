import sys
import os
file_path = os.getcwd()
sys.path.append(file_path)

import root_gnn_base.utils as utils
import argparse
from root_gnn_base.batched_dataset import PreBatchedDataset
from root_gnn_base.batched_dataset import LazyPreBatchedDataset

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', type=str, required=True)
    add_arg('--dataset', type=str, required=True)
    add_arg('--chunk', type=int, default=0)
    add_arg('--shuffle_mode', action='store_true', help='Shuffle the dataset before training.')
    args = parser.parse_args()

    config = utils.load_config(args.config)
    dset_config = config['Datasets'][args.dataset]
    batch_size = config['Training']['batch_size']
    if not args.shuffle_mode:
        dset = utils.buildFromConfig(dset_config, {'process_chunks': [args.chunk,]})
    else:
        dset = utils.buildFromConfig(dset_config)
        if 'batch_size' in dset_config:
            batch_size = dset_config['batch_size']

        shuffle_chunks = dset_config.get('shuffle_chunks', 10)
        padding_mode = dset_config.get('padding_mode', 'STEPS')
        fold_conf = dset_config["folding"]
        print(f"shuffle_chunks = {shuffle_chunks}, args.chunk = {args.chunk}, padding_mode = {padding_mode}")
        if dset_config["class"] == "LazyMultiLabelDataset":
            LazyPreBatchedDataset(start_dataset = dset, batch_size = batch_size, mask_fn = utils.fold_selection(fold_conf, "train"), suffix = utils.fold_selection_name(fold_conf, "train"), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode)
            LazyPreBatchedDataset(start_dataset = dset, batch_size = batch_size, mask_fn = utils.fold_selection(fold_conf, "test"),  suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode)

        else:
            PreBatchedDataset(dset, batch_size, utils.fold_selection(fold_conf, "train"), suffix = utils.fold_selection_name(fold_conf, "train"), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode)
            PreBatchedDataset(dset, batch_size, utils.fold_selection(fold_conf, "test"),  suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, chunkno = args.chunk, padding_mode = padding_mode)

if __name__ == "__main__":
    main()