import dgl
import torch
import numpy as np
import awkward as ak
from dataclasses import dataclass
from typing import List, Any, Dict
from tqdm import tqdm

@dataclass
class ChunkConfig:
    name: str
    label: str
    chunk_id: int
    batch_size: int
    arrays: List[Any]
    particles: List[Any]
    features: List[Any]
    branches: List[Any]
    dtype: torch.dtype
    save_path: str

def process_chunk(cfg: ChunkConfig):
    n_entries = len(cfg.arrays)
    arrays_ordered = {}

    for b in cfg.branches:
        if b in cfg.arrays.fields:
            arrays_ordered[b] = cfg.arrays[b]
        elif b.endswith("_energy"):
            prefix = b[:-7]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            eta_name = f"{prefix}_eta"
            arrays_ordered[b] = cfg.arrays[pt_name] * np.cosh(cfg.arrays[eta_name])
        elif "node_type" in b:
            prefix = b[:-10]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            index = cfg.particles.index(prefix)
            arrays_ordered[b] = ak.ones_like(cfg.arrays[pt_name]) * index
        else:
            prefix = b.split("_")[0]
            pt_name = f"{prefix}_pt"
            if prefix == "MET":
                pt_name = f"{prefix}_met"
            arrays_ordered[b] = ak.zeros_like(cfg.arrays[pt_name])

    graphs = []

    for i in range(n_entries):
        node_features_list = []
        for p in cfg.particles:
            feats = []
            for f in cfg.features:
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
            node_features = np.empty((0, len(cfg.features)))
        graphs.append(make_graph(node_features, dtype=cfg.dtype))
    
    labels = torch.full((len(graphs),), cfg.label, dtype=cfg.dtype)
    save_graphs(f"{cfg.save_path}/{cfg.name}_{cfg.chunk_id:04d}.bin", graphs, {'labels': labels})
    return

def save_graphs(f: str, g: List[dgl.DGLGraph], metadata: Dict) -> None:
    dgl.save_graphs(f, g, metadata)

def load_graphs(f: str):
    g, metadata = dgl.load_graphs(f)
    return g, metadata['labels']

src_dst_cache = {}
def get_src_dst(num_nodes):
    if num_nodes not in src_dst_cache:
        src, dst = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        src_dst_cache[num_nodes] = (src.flatten(), dst.flatten())
    return src_dst_cache[num_nodes]

@torch.jit.script
def compute_edge_features(eta, phi, src, dst):
    deta = eta[src] - eta[dst]
    dphi = phi[src] - phi[dst]
    dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi
    dR = torch.sqrt(deta ** 2 + dphi ** 2)
    edge_features = torch.stack([dR, deta, dphi], dim=1)
    return edge_features

# TODO: normalize all features
def make_graph(node_features: np.array, dtype=torch.float32):
    node_features = torch.tensor(node_features, dtype=dtype)
    num_nodes = node_features.shape[0]
    if num_nodes == 0:
        g = dgl.graph(([], []))
        g.ndata['features'] = node_features
        g.edata['features'] = torch.empty((0, 3), dtype=dtype)
        g.globals = torch.tensor([0], dtype=dtype)
        return g
    src, dst = get_src_dst(num_nodes)
    src = src.flatten()
    dst = dst.flatten()
    g = dgl.graph((src, dst))
    g.ndata['features'] = node_features

    eta = node_features[:, 1]
    phi = node_features[:, 2]
    edge_features = compute_edge_features(eta, phi, src, dst)
    g.edata['features'] = edge_features

    g.globals = torch.tensor([num_nodes], dtype=dtype)
    return g