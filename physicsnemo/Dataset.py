import os
import uproot
import dgl
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from typing import List
from typing import Union
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset

from dgl.dataloading import GraphDataLoader


os.environ["TMPDIR"] = "/pscratch/sd/j/joshuaho/tmp"

def make_graph(node_features: np.array, dtype=torch.float32):
    node_features = torch.tensor(node_features, dtype=dtype)
    num_nodes = node_features.shape[0]
    if num_nodes == 0:
        # Return an empty graph
        g = dgl.graph(([], []))
        g.ndata['features'] = node_features
        g.edata['features'] = torch.empty((0, 3), dtype=dtype)
        g.globals = torch.tensor([0], dtype=dtype)
        return g

    src, dst = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
    src = src.flatten()
    dst = dst.flatten()
    g = dgl.graph((src, dst))
    g.ndata['features'] = node_features  # shape: (num_nodes, num_features)

    # Compute edge features
    eta = node_features[:, 1]
    phi = node_features[:, 2]
    deta = eta[src] - eta[dst]
    dphi = phi[src] - phi[dst]
    dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi
    dR = torch.sqrt(deta ** 2 + dphi ** 2)
    edge_features = torch.stack([deta, dphi, dR], dim=1)
    g.edata['features'] = edge_features

    g.globals = torch.tensor([num_nodes], dtype=dtype)

    return g

def process_chunk(args):
    name, label, chunk_id, arrays, particles, features, branches, dtype, save_path = args
    n_entries = len(arrays)
    arrays_ordered = {}

    arrays_ordered = {}
    for b in branches:
        if b in arrays.fields:
            arrays_ordered[b] = arrays[b]
        elif b.endswith("_energy"):
            prefix = b[:-7]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            eta_name = f"{prefix}_eta"
            arrays_ordered[b] = arrays[pt_name] * np.cosh(arrays[eta_name])
        elif "node_type" in b:
            prefix = b[:-10]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            index = particles.index(prefix)
            arrays_ordered[b] = ak.ones_like(arrays[pt_name]) * index
        else:
            prefix = b.split("_")[0]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            arrays_ordered[b] = ak.zeros_like(arrays[pt_name])

    graphs = []
    for i in range(n_entries):
        if (i % 250 == 0):
            print(f"{name} chunk {chunk_id} processed {i} events")
        node_features_list = []
        for p in particles:
            feats = []
            for f in features:
                branch = f"{p}_{f}"
                if p == "MET" and f == "pt":
                    branch = "MET_met"
                value = ak.to_numpy(arrays_ordered[branch][i])
                feats.append(value)
            if len(feats[0]) == 0:
                continue
            node_array = np.stack(feats, axis=1)
            node_features_list.append(node_array)
        if node_features_list:
            node_features = np.concatenate(node_features_list, axis=0)
        else:
            node_features = np.empty((0, len(features)))
        graphs.append(make_graph(node_features, dtype=dtype))

    labels = torch.full((len(graphs),), label, dtype=torch.long)
    dgl.save_graphs(f"{save_path}/{name}_{chunk_id:02d}.bin", graphs, {'label': labels})
    print(f"Saved {name} chunk {chunk_id:02d} to {save_path}/{name}_{chunk_id:03d}.bin")
    return

class Root_Graph:
    def __init__(
        self,
        name: str,
        label: int,
        load_path: str,
        save_path: str,
        cfg: DictConfig
    ):
        self.name = name
        self.label = label
        self.load_path = load_path
        self.save_path = save_path
        self.data = None

        self.ttree = cfg.ttree
        self.particles = cfg.particles
        self.features = cfg.features
        self.globals = cfg.globals
        self.chunks = cfg.chunks

        self.train_val_test_split = cfg.train_val_test_split
        assert np.sum(self.train_val_test_split) == 1, "train_val_test_split must sum to 1"

        dtype_str = getattr(cfg, "type", "torch.float32")
        if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
            self.dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
        else:
            self.dtype = torch.float32
        print(f"Initializing dataset {name} with dtype {self.dtype}")

    def get_branches(self) -> List[str]:
        branches = [f"{p}_{f}" for p in self.particles for f in self.features]
        branches += self.globals
        branches = ["MET_met" if b == "MET_pt" else b for b in branches]
        return branches

    def process(self, max_workers: int = 32):
        branches = self.get_branches()
        with uproot.open(f"{self.load_path}:{self.ttree}") as tree:
            available_branches = set(tree.keys())
            num_entries = tree.num_entries

        print(f"Getting branches: {branches}")
        graphs = []

        step_size = math.ceil(num_entries / self.chunks)

        # Prepare chunk arguments for each chunk
        chunk_args_list = []
        for chunk_id, arrays in enumerate(
            uproot.iterate(
                f"{self.load_path}:{self.ttree}",
                expressions=[b for b in branches if b in available_branches],
                step_size=step_size,
                library="ak"
            )
        ):
            # Pass everything needed for chunk processing
            chunk_args_list.append((self.name, self.label, chunk_id, arrays, self.particles, self.features, branches, self.dtype, self.save_path))

        # Parallel processing of chunks

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(process_chunk, args) for args in chunk_args_list]
            for future in as_completed(futures):
                future.result()
                
        return
    
    def train_val_test(self):
        split = self.train_val_test_split
        # Collect all graphs and labels first
        all_graphs = []
        all_labels = []
        files = [f"{self.save_path}/{self.name}_{chunk_id:02d}.bin" for chunk_id in range(self.chunks)]
        for f in files:
            graphs, label_dict = dgl.load_graphs(f)
            all_graphs.extend(graphs)
            # Assuming label_dict['label'] is shape (num_graphs_in_file,)
            all_labels.extend(label_dict['label'].tolist())

        n = len(all_graphs)
        rand = np.random.rand(n)
        train_idx = rand < split[0]
        val_idx = (rand >= split[0]) & (rand < split[0] + split[1])
        test_idx = rand >= split[0] + split[1]

        train_graphs = [g for g, flag in zip(all_graphs, train_idx) if flag]
        val_graphs = [g for g, flag in zip(all_graphs, val_idx) if flag]
        test_graphs = [g for g, flag in zip(all_graphs, test_idx) if flag]

        train_labels = [l for l, flag in zip(all_labels, train_idx) if flag]
        val_labels = [l for l, flag in zip(all_labels, val_idx) if flag]
        test_labels = [l for l, flag in zip(all_labels, test_idx) if flag]

        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)

        dgl.save_graphs(f"{self.save_path}/{self.name}_train.bin", train_graphs, {'label': train_labels})
        dgl.save_graphs(f"{self.save_path}/{self.name}_val.bin", val_graphs, {'label': val_labels})
        dgl.save_graphs(f"{self.save_path}/{self.name}_test.bin", test_graphs, {'label': test_labels})

        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    def load(self):
        # List of expected files
        files = [f"{self.save_path}/{self.name}_{chunk_id:02d}.bin" for chunk_id in range(self.chunks)]
        
        # Check if all files exist
        if not all(os.path.exists(f) for f in files):
            print("graphs not found, processing data...")
            self.process()
        else:
            print("graphs found, skipping processing...")
        
        # Check if train/val/test exist:
        files = [f"{self.save_path}/{self.name}_{split}.bin" for split in ["train", "val", "test"]]
        if not all(os.path.exists(f) for f in files):
            print("train/val/test split not found, splitting graphs...")
            self.train_val_test()
        else:
            print("train/val/test split found, skipping splitting...")

        print("loading graphs...")
        train_graphs, train_label_dict = dgl.load_graphs(f"{self.save_path}/{self.name}_train.bin")
        val_graphs, val_label_dict = dgl.load_graphs(f"{self.save_path}/{self.name}_val.bin")
        test_graphs, test_label_dict = dgl.load_graphs(f"{self.save_path}/{self.name}_test.bin")

        train_labels = train_label_dict['label']
        val_labels = val_label_dict['label']
        test_labels = test_label_dict['label']

        return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

def get_dataset(cfg: DictConfig):

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    all_train_graphs = []
    all_train_labels = []
    all_val_graphs = []
    all_val_labels = []
    all_test_graphs = []
    all_test_labels = []

    for ds in cfg.datasets:
        name = ds['name']
        load_path = ds.get('load_path', f"{cfg.paths.data_dir}/{name}.root")
        save_path = ds.get('save_path', f"{cfg.paths.save_dir}/")
        graph = Root_Graph(name, ds.get('label'), load_path, save_path, cfg.root_graph)
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = graph.load()
        all_train_graphs.extend(train_graphs)
        all_train_labels.extend(train_labels)
        all_val_graphs.extend(val_graphs)
        all_val_labels.extend(val_labels)
        all_test_graphs.extend(test_graphs)
        all_test_labels.extend(test_labels)

    train_dataset = GraphDataset(all_train_graphs, all_train_labels)
    val_dataset = GraphDataset(all_val_graphs, all_val_labels)
    test_dataset = GraphDataset(all_test_graphs, all_test_labels)

    batch_size = cfg.root_graph.batch_size

    train_loader = GraphDataLoader(train_dataset, {'batch_size' : batch_size, 'shuffle' : True})
    val_loader = GraphDataLoader(val_dataset, {'batch_size' : batch_size, 'shuffle' : False})
    test_loader = GraphDataLoader(test_dataset, {'batch_size' : batch_size, 'shuffle' : False})

    return train_loader, val_loader, test_loader