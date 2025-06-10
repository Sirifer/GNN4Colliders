import sys
import os
file_path = os.getcwd()
sys.path.append(file_path)

import argparse
import yaml

import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import SubsetRandomSampler, SequentialSampler


def my_error_handler(level, abort, location, msg):
    # Log the error message to a file instead of printing
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"Error in {location}: {msg}\n")

    # Optionally, print the error message to the console
    # print(f"Error in {location}: {msg}")

    # Decide whether to abort based on the error level
    if abort:
        raise RuntimeError(f"Fatal error in {location}: {msg}")

class CustomPreBatchedDataset(DGLDataset):
    def __init__(self, start_dataset, batch_size, mask_fn=None, drop_last=False, shuffle=False, **kwargs):
        self.start_dataset = start_dataset
        self.batch_size = batch_size
        self.mask_fn = mask_fn or (lambda x: torch.ones(len(x), dtype=torch.bool))
        self.drop_last = drop_last
        self.shuffle = shuffle
        super().__init__(name=start_dataset.name + '_custom_prebatched', save_dir=start_dataset.save_dir)

    def process(self):
        mask = self.mask_fn(self.start_dataset)
        indices = torch.arange(len(self.start_dataset))[mask]
        print(f"Number of elements after masking: {len(indices)}")  # Debugging print

        if self.shuffle:
            sampler = SubsetRandomSampler(indices)
        else:
            sampler = SequentialSampler(indices)

        self.dataloader = GraphDataLoader(
            self.start_dataset, 
            sampler=sampler, 
            batch_size=self.batch_size, 
            drop_last=self.drop_last
        )
        print(f"Batch size set in DataLoader: {self.batch_size}")  # Debugging print

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        sampler = SequentialSampler(idx)
        dloader = GraphDataLoader(self.start_dataset, sampler=sampler, batch_size=self.batch_size, drop_last=False)
        return next(iter(dloader))

    def __len__(self):
        return len(self.start_dataset)

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

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', type=str, required=True)
    add_arg('--target', type=str, required=True)
    add_arg('--destination', type=str, default='')
    add_arg('--chunkno', type=int, default=0)
    add_arg('--chunks', type=int, default=1)
    add_arg('--write', action='store_true')
    add_arg('--ckpt', type=int, default=-1)
    add_arg('--clobber', action='store_true')
    add_arg('--tree', type=str, default='')
    add_arg('--branch_name', type=str, default='score')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.destination == '':
        args.destination = os.path.join(config['Training_Directory'], 'inference/', os.path.split(args.target)[1])
    else:
        args.destination = args.destination
    if not args.write:
        args.destination = args.destination.replace('.root', '') + f'_chunk{args.chunkno}.npz'

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
    dset_config = config['Datasets'][list(config['Datasets'].keys())[0]]
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
    loader = CustomPreBatchedDataset(dset, batch_size)
    loader.process()
    # loader = dataset.PreBatchedDataset(dset, batch_size, shuffle=False, drop_last=False, save_to_disk=False, chunks = 1, num_workers=0)
    lend = time.time()
    print('Loader finished in {:.2f} seconds'.format(lend - lstart))
    sample_graph, _, _, global_sample = loader[0]

    print('dset length =', len(dset))
    print('loader length =', len(loader))

    model = utils.buildFromConfig(config['Model'], {'sample_graph' : sample_graph, 'sample_global': global_sample}).to(device)
    if args.ckpt < 0:
        ep, checkpoint = utils.get_last_epoch(config, args.ckpt, device=device)
    else:
        ep, checkpoint = utils.get_specific_epoch(config, args.ckpt, device=device)
    #Bad filler for models which were compiled. Have to remove this prefix.
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
        # scores.append(finish_fn(pred).detach().cpu().numpy())
        if (finish_fn.__class__.__name__ == "ContrastiveClusterFinish"):
            scores.append(pred.detach().cpu().numpy())
        else:
            scores.append(finish_fn(pred).detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        tracking_info.append(track.detach().cpu().numpy())

    # for batch, label, track, globals in loader:
    #     batch = batch.to(device)
    #     pred = model(batch, globals.to(device))
    #     print(f'Batch size: {batch.batch_size if hasattr(batch, "batch_size") else "Unavailable"}')
    #     print(f'Prediction shape: {pred.shape}')
    #     ibatch += 1
    #     scores.append(finish_fn(pred).detach().cpu().numpy())
    #     labels.append(label.detach().cpu().numpy())
    #     tracking_info.append(track.detach().cpu().numpy())
    #     exit()

    score_size = scores[0].shape[1]
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    tracking_info = np.concatenate(tracking_info)
    end = time.time()
    
    print('Inference finished in {:.2f} seconds'.format(end - start))

    if args.write:
        # ROOT.SetErrorHandler(my_error_handler)
        ROOT.gErrorIgnoreLevel = ROOT.kFatal
        # ROOT.gSystem.RedirectOutput("/dev/null", "w")
        
        # Open the original ROOT file
        infile = ROOT.TFile.Open(args.target)
        tree = infile.Get(dset_config['args']['tree_name'])

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.split(args.destination)[0], exist_ok=True)

        # Create a new ROOT file to write the modified tree
        outfile = ROOT.TFile.Open(args.destination, 'RECREATE')

        # Clone the original tree, including data
        outtree = tree.CloneTree(0)  # Clone all entries

        # Determine if scores is a list of single values or vectors
        from ROOT import std
        if isinstance(scores[0], (list, tuple, np.ndarray)):  # Check if scores contains vectors
            # Create a new branch for scores as a vector of floats
            scores_branch_vec = std.vector('float')()
            outtree.Branch(args.branch_name, scores_branch_vec)
            is_vector = True
        else:  # Scores contains single values
            # Create a new branch for scores as a single float
            score_branch_arr = array('f', [0])
            outtree.Branch(args.branch_name, score_branch_arr, f'{args.branch_name}/F')
            is_vector = False

        # Write scores to the new branch
        print(f'Writing {len(scores)} scores to tree')

        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            
            if is_vector:
                # Clear the vector
                scores_branch_vec.clear()
                
                # Add all elements from scores[i] to the vector
                for value in scores[i]:
                    scores_branch_vec.push_back(float(value))  # Use push_back to add elements one by one
            else:
                # Fill the score branch with the current single score
                score_branch_arr[0] = float(scores[i])  # Ensure the value is a float

            # Fill the output tree with all branches, including the new scores branch
            outtree.Fill()

        # Write the modified tree to the new file
        print(f'Writing to file {args.destination}')
        print(f'Input entries: {tree.GetEntries()}, Output entries: {outtree.GetEntries()}')
        outtree.Write()
        outfile.Close()
        infile.Close()
    else:
        os.makedirs(os.path.split(args.destination)[0], exist_ok=True)
        np.savez(args.destination, scores=scores, labels=labels, tracking_info=tracking_info)

if __name__ == '__main__':
    main()



