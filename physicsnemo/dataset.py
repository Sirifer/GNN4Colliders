import os
import uproot
import dgl
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import dataset_utils as utils

from torch.utils.data import Dataset

from dgl.dataloading import GraphDataLoader

class RootDataset:
    def __init__(
        self,
        name: str,
        label: int,
        load_path: str,
        save_path: str,
        device: str,
        cfg: DictConfig
    ):
        self.name = name
        self.label = label
        self.load_path = load_path
        self.save_path = save_path
        self.data = None
        self.device = device

        self.ttree = cfg.ttree
        self.particles = cfg.particles
        self.features = cfg.features
        self.globals = cfg.globals
        self.step_size = cfg.step_size
        self.batch_size = cfg.batch_size

        self.train_val_test_split = cfg.train_val_test_split
        assert np.sum(self.train_val_test_split) == 1, "train_val_test_split must sum to 1"

        dtype_str = getattr(cfg, "type", "torch.float32")
        if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
            self.dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
        else:
            self.dtype = torch.float32
        print(f"initializing dataset {name} with dtype {self.dtype}")

    def get_branches(self) -> List[str]:
        branches = [f"{p}_{f}" for p in self.particles for f in self.features]
        branches += self.globals
        branches = ["MET_met" if b == "MET_pt" else b for b in branches]
        return branches

    def process(self):
        branches = self.get_branches()
        with uproot.open(f"{self.load_path}:{self.ttree}") as tree:
            available_branches = set(tree.keys())
            num_entries = tree.num_entries

        print(f"getting branches: {branches}")
        
        num_cpus = os.cpu_count()
        total_chunks = np.ceil(num_entries / self.step_size)

        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = []
            for chunk_id, arrays in enumerate(
                tqdm(
                    uproot.iterate(
                        f"{self.load_path}:{self.ttree}",
                        expressions=[b for b in branches if b in available_branches],
                        step_size=self.step_size,
                        library="ak"
                    ),
                    desc="loading root file",
                    total=total_chunks,
                    position=0,
                    leave=False
                )
            ):
                    
                cfg = utils.ChunkConfig(
                    name=self.name,
                    label=self.label,
                    chunk_id=chunk_id,
                    batch_size=self.batch_size,
                    arrays=arrays,
                    particles=self.particles,
                    features=self.features,
                    branches=branches,
                    dtype=self.dtype,
                    save_path=self.save_path,
                )

                futures.append(executor.submit(utils.process_chunk, cfg))

            
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                import traceback
                print("Exception in worker process:")
                traceback.print_exception(type(e), e, e.__traceback__)
        return
    
    def load(self):
        with uproot.open(f"{self.load_path}:{self.ttree}") as tree:
            num_entries = tree.num_entries
        total_chunks = int(np.ceil(num_entries / self.step_size))
        
        chunk_files = [f"{self.save_path}/{self.name}_{chunk_id:04d}.bin" for chunk_id in range(total_chunks)]
        if not all(os.path.exists(f) for f in chunk_files):
            print("graphs not found. processing root file...")
            self.process()

        graph_tuple_list = []
        
        for chunk_id, f in enumerate(chunk_files):
            if chunk_id < total_chunks - 1:
                n_graphs = self.step_size
            else:
                n_graphs = num_entries - self.step_size * (total_chunks - 1)
            graph_tuple_list.extend((f, idx) for idx in range(n_graphs))
        
        split = self.train_val_test_split
        n_total = len(graph_tuple_list)
        n_train = int(split[0] * n_total)
        n_val = int(split[1] * n_total)

        train_tuples = graph_tuple_list[:n_train]
        val_tuples   = graph_tuple_list[n_train:n_train + n_val]
        test_tuples  = graph_tuple_list[n_train + n_val:]
        return train_tuples, val_tuples, test_tuples

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    def shuffle(self):
        # TODO: implement graph shuffling
        return self.graphs

class GraphTupleDataset:
    def __init__(self, tuple_list):
        self.tuples = tuple_list
    def __len__(self):
        return len(self.tuples)
    def __getitem__(self, idx):
        filepath, graph_idx = self.tuples[idx]
        graphs, labels = utils.load_graphs(filepath)
        return graphs[graph_idx], labels[graph_idx]
    
def get_dataset(cfg: DictConfig, device):

    all_train = []
    all_val = []
    all_test = []

    for ds in cfg.datasets:
        name = ds['name']
        load_path = ds.get('load_path', f"{cfg.paths.data_dir}/{name}.root")
        save_path = ds.get('save_path', f"{cfg.paths.save_dir}/")
        datastet = RootDataset(name, ds.get('label'), load_path, save_path, device, cfg.root_dataset)
        train, val, test = datastet.load()
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    train_dataset = GraphTupleDataset(all_train)
    val_dataset = GraphTupleDataset(all_val)
    test_dataset = GraphTupleDataset(all_test)

    batch_size = cfg.root_dataset.batch_size
    
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader   = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader  = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    print("all data loaded successfully")
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader