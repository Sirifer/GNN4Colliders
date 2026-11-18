import sys
import os
import argparse
import yaml
import gc
from array import array
import fnmatch
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import SubsetRandomSampler, SequentialSampler

class CustomPreBatchedDataset(DGLDataset):
    def __init__(self, start_dataset, batch_size, chunkno=0, chunks=1, mask_fn=None, drop_last=False, shuffle=False, **kwargs):
        self.start_dataset = start_dataset
        self.batch_size = batch_size
        self.mask_fn = mask_fn or (lambda x: torch.ones(len(x), dtype=torch.bool))
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.chunkno = chunkno
        self.chunks = chunks
        super().__init__(name=start_dataset.name + '_custom_prebatched', save_dir=start_dataset.save_dir)

    def process(self):
        mask = self.mask_fn(self.start_dataset)
        indices = torch.arange(len(self.start_dataset))[mask]
        print(f"Number of elements after masking: {len(indices)}")  # Debugging print

        # --- CHUNK SPLITTING ---
        total = len(indices)
        if self.chunks == 1:
            chunk_indices = indices
            print(f"Chunks=1, using all {total} indices.")
        else:
            chunk_size = (total + self.chunks - 1) // self.chunks
            start = self.chunkno * chunk_size
            end = min((self.chunkno + 1) * chunk_size, total)
            chunk_indices = indices[start:end]
            print(f"Working on chunk {self.chunkno}/{self.chunks}: indices {start}:{end} (total {len(chunk_indices)})")

        if self.shuffle:
            sampler = SubsetRandomSampler(chunk_indices)
        else:
            sampler = SequentialSampler(chunk_indices)

        self.dataloader = GraphDataLoader(
            self.start_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            drop_last=self.drop_last
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        sampler = SequentialSampler(idx)
        dloader = GraphDataLoader(self.start_dataset, sampler=sampler, batch_size=self.batch_size, drop_last=False)
        return next(iter(dloader))

    def __len__(self):
        mask = self.mask_fn(self.start_dataset)
        indices = torch.arange(len(self.start_dataset))[mask]
        total = len(indices)
        if self.chunks == 1:
            return total
        chunk_size = (total + self.chunks - 1) // self.chunks
        start = self.chunkno * chunk_size
        end = min((self.chunkno + 1) * chunk_size, total)
        return end - start

def include_config(conf):
    if 'include' in conf:
        for i in conf['include']:
            with open(i) as f:
                conf.update(yaml.load(f, Loader=yaml.FullLoader))
        del conf['include']

def load_config(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    include_config(conf)
    return conf

def _branch_names(tree):
    return {branch.GetName() for branch in tree.GetListOfBranches()}

def _compare(value, cut, op):
    if op == '>':
        return value > cut
    if op == '>=':
        return value >= cut
    if op == '<':
        return value < cut
    if op == '<=':
        return value <= cut
    if op == '==':
        return value == cut
    if op == '!=':
        return value != cut
    raise ValueError(f'Unknown selection operator: {op}')

def compute_selection_pass(tree, selections):
    if not selections:
        return 1

    names = _branch_names(tree)

    for selection in selections:
        if isinstance(selection, str):
            tokens = set(re.findall(r'\b[A-Za-z_]\w*\b', selection))
            needed = tokens & names

            context = {}
            for name in needed:
                context[name] = getattr(tree, name)

            try:
                passed = bool(eval(selection, {"__builtins__": {}}, context))
            except NameError as exc:
                raise NameError(
                    f"Selection expression references an unknown name: {selection}"
                ) from exc

            if not passed:
                return 0
        else:
            if not isinstance(selection, (tuple, list)) or len(selection) != 3:
                raise ValueError(
                    f"Selection must be a string or a 3-item sequence "
                    f"(var, cut, op), got: {selection}"
                )

            var, cut, op = selection

            if var not in names:
                raise KeyError(f"Selection references missing branch: {var}")

            value = getattr(tree, var)

            if not _compare(value, cut, op):
                return 0

    return 1

def select_dataset_config(configs, target_path):
    target_name = os.path.basename(target_path)
    for config in configs:
        dset_config = config['Datasets'][list(config['Datasets'].keys())[0]]
        file_names = dset_config['args'].get('file_names', '')
        patterns = file_names if isinstance(file_names, list) else [file_names]
        for pattern in patterns:
            if fnmatch.fnmatch(target_name, os.path.basename(pattern)):
                return dset_config
    return configs[0]['Datasets'][list(configs[0]['Datasets'].keys())[0]]

def main():

    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', type=str, nargs='+', required=True, help="List of config files")
    add_arg('--target', type=str, required=True)
    add_arg('--destination', type=str, default='')
    add_arg('--chunkno', type=int, default=0)
    add_arg('--chunks', type=int, default=1)
    add_arg('--write', action='store_true')
    add_arg('--ckpt', type=int, default=-1)
    add_arg('--var', type=str, default='Test_AUC')
    add_arg('--mode', type=str, default='max')
    add_arg('--clobber', action='store_true')
    add_arg('--tree', type=str, default='')
    add_arg('--branch_name', type=str, nargs='+', required=True, help="List of branch names corresponding to configs")
    args = parser.parse_args()

    if(len(args.config) != len(args.branch_name)):
        print(f"configs and branch names do not match")
        return

    config = load_config(args.config[0])

    # --- OUTPUT DESTINATION LOGIC ---
    if args.destination == '':
        base_dest = os.path.join(config['Training_Directory'], 'inference/', os.path.split(args.target)[1])
    else:
        base_dest = args.destination

    base_dest = base_dest.replace('.root', '').replace('.npz', '')
    if args.chunks > 1:
        chunked_dest = f"{base_dest}_chunk{args.chunkno}"
    else:
        chunked_dest = base_dest
    chunked_dest += '.root' if args.write else '.npz'
    args.destination = chunked_dest

    # --- FILE EXISTENCE CHECK ---
    if os.path.exists(args.destination):
        print(f'File {args.destination} already exists.')
        if args.clobber:
            print('Clobbering.')
        else:
            print('Exiting.')
            return
    else:
        print(f'Writing to {args.destination}')

    import time
    start  = time.time()
    import ROOT
    import torch
    from array import array
    import numpy as np
    from root_gnn_base import batched_dataset as dataset
    from root_gnn_base import utils
    end = time.time()
    print('Imports finished in {:.2f} seconds'.format(end - start))

    start = time.time()
    dset_config = select_dataset_config([load_config(c) for c in args.config], args.target)
    if dset_config['class'] == 'LazyDataset':
        dset_config['class'] = 'EdgeDataset'
    elif dset_config['class'] == 'LazyMultiLabelDataset':
        dset_config['class'] = 'MultiLabelDataset'
    elif dset_config['class'] == 'PhotonIDDataset':
        dset_config['class'] = 'UnlazyPhotonIDDataset'
    elif dset_config['class'] == 'kNNDataset':
        dset_config['class'] = 'UnlazyKNNDataset'
    dset_config['args']['raw_dir'] = os.path.split(args.target)[0]
    dset_config['args']['file_names'] = os.path.split(args.target)[1]
    dset_config['args']['save'] = False
    dset_config['args']['chunks'] = args.chunks
    dset_config['args']['process_chunks'] = [args.chunkno,]
    dset_config['args']['selections'] = []

    dset_config['args']['save_dir'] = os.path.dirname(args.destination)

    if args.tree != '':
        dset_config['args']['tree_name'] = args.tree

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dstart = time.time()
    dset = utils.buildFromConfig(dset_config)
    dend = time.time()
    print('Dataset finished in {:.2f} seconds'.format(dend - dstart))

    print(dset)

    batch_size = config['Training']['batch_size']
    lstart = time.time()
    loader = CustomPreBatchedDataset(
        dset,
        batch_size,
        chunkno=args.chunkno,
        chunks=args.chunks
    )
    loader.process()
    lend = time.time()
    print('Loader finished in {:.2f} seconds'.format(lend - lstart))
    sample_graph, _, _, global_sample = loader[0]
    global_sample = []

    print('dset length =', len(dset))
    print('loader length =', len(loader))

    all_scores = {}
    all_labels = {}
    all_tracking = {}
    with torch.no_grad():
        for config_file, branch in zip(args.config, args.branch_name):
            config = load_config(config_file)
            model = utils.buildFromConfig(config['Model'], {'sample_graph' : sample_graph, 'sample_global': global_sample}).to(device)
            
            if args.ckpt < 0:
                ep, checkpoint = utils.get_best_epoch(config, var=args.var, mode=args.mode, device=device)
            else:
                ep, checkpoint = utils.get_specific_epoch(config, args.ckpt, device=device)
            # Remove distributed/compiled prefixes if present
            mds_copy = {}
            for key in checkpoint['model_state_dict'].keys():
                newkey = key.replace('module.', '')
                newkey = newkey.replace('_orig_mod.', '')
                mds_copy[newkey] = checkpoint['model_state_dict'][key]
            model.load_state_dict(mds_copy)
            model.eval()

            end = time.time()
            print('Model and dataset finished in {:.2f} seconds'.format(end - start))
            print('Starting inference')
            start = time.time()

            finish_fn = torch.nn.Sigmoid()
            if 'Loss' in config:
                finish_fn = utils.buildFromConfig(config['Loss']['finish'])

            scores = []
            labels = []
            tracking_info = []
            ibatch = 0

            for batch, label, track, globals in loader.dataloader:
                batch = batch.to(device)
                pred = model(batch, globals.to(device))
                ibatch += 1
                if (finish_fn.__class__.__name__ == "ContrastiveClusterFinish"):
                    scores.append(pred.detach().cpu().numpy())
                else:
                    scores.append(finish_fn(pred).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
                tracking_info.append(track.detach().cpu().numpy())

            score_size = scores[0].shape[1] if len(scores[0].shape) > 1 else 1
            scores = np.concatenate(scores)
            labels = np.concatenate(labels)
            tracking_info = np.concatenate(tracking_info)
            end = time.time()

            print('Inference finished in {:.2f} seconds'.format(end - start))
            all_scores[branch] = scores
            all_labels[branch] = labels
            all_tracking[branch] = tracking_info

    if args.write:
        from ROOT import std
        write_config = select_dataset_config([load_config(c) for c in args.config], args.target)
        # Open the original ROOT file
        infile = ROOT.TFile.Open(args.target)
        tree = infile.Get(write_config['args']['tree_name'])
        selections = write_config.get('selections', [])

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.split(args.destination)[0], exist_ok=True)

        # Create a new ROOT file to write the modified tree
        outfile = ROOT.TFile.Open(args.destination, 'RECREATE')

        # Clone the original tree structure
        outtree = tree.CloneTree(0)

        # Create branches for all scores
        branch_vectors = {}
        for branch, scores in all_scores.items():
            if isinstance(scores[0], (list, tuple, np.ndarray)) and len(scores[0]) > 1:
                # Create a new branch for vectors
                branch_vectors[branch] = std.vector('float')()
                outtree.Branch(branch, branch_vectors[branch])
            else:
                # Create a new branch for single floats
                branch_vectors[branch] = array('f', [0])
                outtree.Branch(branch, branch_vectors[branch], f'{branch}/F')

        selection_pass = array('i', [1])
        outtree.Branch('selection_pass', selection_pass, 'selection_pass/I')

        # Fill the tree
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            selection_pass[0] = compute_selection_pass(tree, selections)

            for branch, scores in all_scores.items():
                branch_data = branch_vectors[branch]
                if isinstance(branch_data, array):  # Check if it's a single float array
                    branch_data[0] = float(scores[i])
                else:  # Assume it's a std::vector<float>
                    branch_data.clear()
                    for value in scores[i]:
                        branch_data.push_back(float(value))

            outtree.Fill()

        # Write the modified tree to the new file
        print(f'Writing to file {args.destination}')
        print(f'Input entries: {tree.GetEntries()}, Output entries: {outtree.GetEntries()}')
        print(f'Wrote scores to {args.branch_name} and selection_pass')
        outtree.Write()
        outfile.Close()
        infile.Close()
    else:
        os.makedirs(os.path.split(args.destination)[0], exist_ok=True)
        np.savez(args.destination, scores=all_scores, labels=all_labels, tracking_info=all_tracking)

if __name__ == '__main__':
    main()
