import os
import uproot
import dgl
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from typing import List
from concurrent.futures import ThreadPoolExecutor

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

class Root_Graph:
    def __init__(
        self,
        name: str,
        load_path: str,
        save_path: str,
        cfg: DictConfig
    ):
        self.name = name
        self.load_path = load_path
        self.save_path = save_path
        self.data = None

        self.ttree = cfg.ttree
        self.particles = cfg.particles
        self.features = cfg.features
        self.globals = cfg.globals
        self.batch_size = cfg.batch_size

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

    def process(self, max_workers: int = 16):
        branches = self.get_branches()
        # Get available branches from the tree
        with uproot.open(f"{self.load_path}:{self.ttree}") as tree:
            available_branches = set(tree.keys())

        print(f"Getting branches: {branches}")
        graphs = []

        for arrays in uproot.iterate(
            f"{self.load_path}:{self.ttree}",
            expressions=[b for b in branches if b in available_branches],
            step_size=self.batch_size,
            library="ak"
        ):
            n_entries = len(arrays)
            # Convert all arrays to numpy at once, avoid repeated conversion
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
                    index = self.particles.index(prefix)
                    arrays_ordered[b] = ak.ones_like(arrays[pt_name]) * index
                else:
                    prefix = b.split("_")[0]
                    pt_name = f"{prefix}_pt"
                    if prefix == "MET":
                        pt_name = f"{prefix}_met"
                    arrays_ordered[b] = ak.zeros_like(arrays[pt_name])

            def build_graph(i):
                node_features_list = []
                for p in self.particles:
                    feats = []
                    for f in self.features:
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
                    node_features = np.empty((0, len(self.features)))
                return make_graph(node_features, dtype=self.dtype)

            # Parallel graph construction within the chunk
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_graphs = list(executor.map(build_graph, range(n_entries)))
            graphs.extend(chunk_graphs)
            print(f"Processed chunk with {n_entries} entries")

        batched_graph = dgl.batch(graphs)
        print(f"Batched graph: {batched_graph}")
        return batched_graph

    def load(self):
        # Check if processed file exists, if not, process and save it
        if not os.path.exists(self.save_path):
            print(f"{self.save_path} not found, processing data...")
            batched_graphs = self.process()
            # TODO: Save processed graph to self.save_path
            # Example: torch.save(batched_graph, self.save_path)
        # TODO: Load the processed graph from self.save_path
        print(f"Loaded {self.name} from {self.save_path}")

def process_data(cfg: DictConfig):
    graphs = []
    for ds in cfg.datasets:
        name = ds['name']
        load_path = ds.get('load_path', f"{cfg.paths.data_dir}/{name}.root")
        save_path = ds.get('save_path', f"{cfg.paths.save_dir}/{name}_processed.pt")
        graph = Root_Graph(name, load_path, save_path, cfg.root_graph)
        graph.load()
        graphs.append(graph)
    return graphs