import os
import uproot
import dgl
import torch
import numpy as np
from omegaconf import DictConfig
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from dataset import GraphBuilder
from dataset import Graphs
from dataset import Normalization

from dgl.dataloading import GraphDataLoader

class Dataset:
    def __init__(
        self,
        name: str,
        label: int,
        load_path: str,
        save_path: str,
        dtype: torch.dtype,
        device: str,
        cfg: DictConfig
    ):
        self.name = name
        self.label = label
        self.load_path = load_path
        self.save_path = save_path
        self.dtype = dtype
        self.data = None
        self.device = device

        self.ttree = cfg.ttree
        self.features = cfg.features
        self.weights = cfg.weights
        self.globals = cfg.globals
        self.tracking = cfg.tracking
        self.step_size = cfg.step_size
        self.batch_size = cfg.batch_size

        self.prebatch = cfg.get('prebatch', {'enabled': False})

        self.train_val_test_split = cfg.train_val_test_split
        assert np.sum(self.train_val_test_split) == 1, "train_val_test_split must sum to 1"

        print(f"initializing dataset {name} with dtype {self.dtype}")

    def get_branches(self) -> List[str]:
        node_branches = [
            branches
            for particle in self.features.values()
            for branches in particle
            if isinstance(branches, str) and (branches != "CALC_E" or branches != "NUM_NODES")
        ]
        global_branches = [x for x in self.globals if isinstance(x, str)]
        weight_branch = [self.weights] if isinstance(self.weights, str) else []
        tracking_branches = [x for x in self.tracking if isinstance(x, str)]
        label_branch = [self.label] if isinstance(self.label, str) else []

        return node_branches + global_branches + weight_branch + tracking_branches + label_branch

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

            with tqdm(
                uproot.iterate(
                        f"{self.load_path}:{self.ttree}",
                        expressions=[b for b in branches if b in available_branches],
                        step_size=self.step_size,
                        library="ak"
                    ),
                desc="loading root file",
                total=total_chunks,
                position=0,
                leave=True
            ) as pbar:
                
                for chunk_id, arrays in enumerate(pbar):

                    cfg = GraphBuilder.ChunkConfig(
                        name=self.name,
                        label=self.label,
                        chunk_id=chunk_id,
                        batch_size=self.batch_size,
                        arrays=arrays,
                        features=self.features,
                        globals=self.globals,
                        tracking=self.tracking,
                        weights=self.weights,
                        branches=branches,
                        dtype=self.dtype,
                        save_path=self.save_path,
                        prebatch = self.prebatch,
                    )
                    
                    futures.append(executor.submit(GraphBuilder.process_chunk, cfg))

        for idx, future in enumerate(as_completed(futures)):
            try:
                future.result()
            except Exception as e:
                import traceback
                print(f"exception in chunk: {idx}")
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
                if (self.prebatch.enabled):
                    n_graphs = self.step_size // self.prebatch.chunk_size
                else:
                    n_graphs = self.step_size
            else:
                if (self.prebatch.enabled):
                    n_graphs = (num_entries - self.step_size * (total_chunks - 1)) // self.prebatch.chunk_size + 1
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
    
class GraphTupleDataset:
    def __init__(self, tuple_list, stats):
        self.tuple_list = tuple_list
        self.stats = stats
        self.cache = {}

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        f, graph_idx = self.tuple_list[idx]
        if f in self.cache:
            g = self.cache[f]
        else: 
            g = Graphs.load_graphs(f)
            g.normalize(self.stats)
            self.cache[f] = g
        return g[graph_idx]

    @staticmethod
    def collate_fn(samples):
        all_graphs = []
        all_metadata = {}

        # Initialize keys in all_metadata from the first sample
        for k in samples[0][1]:
            all_metadata[k] = []

        for graph, metadata in samples:
            all_graphs.append(graph)
            for k, v in metadata.items():
                all_metadata[k].append(v)

        # Stack or concatenate metadata for each key
        for k in all_metadata:
            # If v is a tensor, stack or cat as appropriate
            # Use torch.cat if v is already [N, ...] (e.g. labels, features)
            # Use torch.stack if v is scalar or needs new dimension
            try:
                all_metadata[k] = torch.cat(all_metadata[k], dim=0)
            except Exception:
                all_metadata[k] = torch.stack(all_metadata[k], dim=0)

        batched_graph = dgl.batch(all_graphs)
        return batched_graph, all_metadata
        
def get_dataset(cfg: DictConfig, device):

    all_train = []
    all_val = []
    all_test = []

    dtype_str = getattr(cfg.root_dataset, "dtype", "torch.float32")
    if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
        dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
    else:
        dtype = torch.float32

    for ds in cfg.datasets:
        name = ds['name']
        load_path = ds.get('load_path', f"{cfg.paths.data_dir}/{name}.root")
        save_path = ds.get('save_path', f"{cfg.paths.save_dir}/")
        datastet = Dataset(name, ds.get('label'), load_path, save_path, dtype, device, cfg.root_dataset)
        train, val, test = datastet.load()
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    stats = Normalization.global_stats(f"{cfg.paths.save_dir}/stats/", dtype=dtype)

    train_dataset = GraphTupleDataset(all_train, stats)
    val_dataset = GraphTupleDataset(all_val, stats)
    test_dataset = GraphTupleDataset(all_test, stats)

    if (cfg.root_dataset.get('prebatch', False)):
        batch_size = cfg.root_dataset.batch_size // cfg.root_dataset.prebatch.chunk_size
        collate_fn = GraphTupleDataset.collate_fn
    else:
        batch_size = cfg.root_dataset.batch_size
        collate_fn = None
    
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5, drop_last=False, collate_fn=collate_fn)
    val_loader   = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=5, drop_last=False, collate_fn=collate_fn)
    test_loader  = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=collate_fn)

    print("all data loaded successfully")
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader