import dgl
import torch
import numpy as np
import awkward as ak
from dataclasses import dataclass
from typing import List, Any, Union

from dataset.Graphs import Graphs, save_graphs
from dataset import Normalization

@dataclass
class ChunkConfig:
    name:           str
    label:          Union[str, int]
    chunk_id:       int
    batch_size:     int
    arrays:         List[Any]
    features:       List[Any]
    globals:        List[Any]
    weights:        Union[str, float]
    tracking:       List[Any]
    branches:       List[Any]
    dtype:          torch.dtype
    save_path:      str
    prebatch:       dict

def process_chunk(cfg: ChunkConfig):
    # Collect everything as lists first
    graph_list = []
    meta_dict = {
        'globals': [],
        'label': [],
        'weight': [],
        'tracking': [],
        'batch_num_nodes': [],
        'batch_num_edges': [],
    }

    for i in range(len(cfg.arrays)):
        g, meta = process_single_entry(cfg, i)
        graph_list.append(g)
        for k in meta_dict:
            meta_dict[k].append(meta[k])

    # Stack all metadata fields into tensors
    for k in meta_dict:
        meta_dict[k] = torch.stack(meta_dict[k])

    graphs = Graphs(graphs=graph_list, metadata=meta_dict)
    Normalization.save_stats(graphs, f"{cfg.save_path}/stats/{cfg.name}_{cfg.chunk_id:04d}.json")

    if getattr(cfg.prebatch, "enabled", False):
        graphs.shuffle()
        graphs.batch(cfg.prebatch["chunk_size"])

    save_graphs(graphs, f"{cfg.save_path}/{cfg.name}_{cfg.chunk_id:04d}.bin")

def process_single_entry(cfg, i):
    # 1) node features
    node_features: List[torch.Tensor] = []

    for particle, branch_list in cfg.features.items():
        feature_tensors: List[torch.Tensor] = []
        for branch in branch_list:
            if branch == "CALC_E":
                pT  = feature_tensors[0]
                eta = feature_tensors[1]
                val = pT * torch.cosh(eta)
            elif isinstance(branch, str):
                arr = cfg.arrays[branch][i]
                val = torch.from_numpy(ak.to_numpy(arr)).to(cfg.dtype)
            else:
                length = feature_tensors[0].shape[0]
                val = torch.full((length,), float(branch), dtype=cfg.dtype)
            feature_tensors.append(val)

        if feature_tensors and feature_tensors[0].numel() > 0:
            block = torch.stack(feature_tensors, dim=1)
            node_features.append(block)

    node_features = torch.cat(node_features, dim=0) if node_features else torch.empty((0, len(cfg.features)), dtype=cfg.dtype)

    # 2) global features
    global_feat_list: List[torch.Tensor] = []
    for b in cfg.globals:
        if b == "NUM_NODES":
            global_feat_list.append(torch.tensor([len(node_features)], dtype=cfg.dtype))
        else:
            arr = cfg.arrays[b][i]
            global_feat_list.append(torch.from_numpy(ak.to_numpy(arr)).to(cfg.dtype))
    global_feat = torch.cat(global_feat_list, dim=0) if global_feat_list else torch.zeros((1,), dtype=cfg.dtype)

    # 3) tracking
    tracking_list: List[torch.Tensor] = []
    for b in cfg.tracking:
        arr = cfg.arrays[b][i]
        tracking_list.append(torch.from_numpy(ak.to_numpy(arr)).to(cfg.dtype))
    tracking = torch.cat(tracking_list, dim=0) if tracking_list else torch.zeros((1,), dtype=cfg.dtype)

    # 4) weight
    weight = float(cfg.arrays[cfg.weights][i]) if isinstance(cfg.weights, str) else cfg.weights
    weight = torch.tensor(weight, dtype=cfg.dtype)

    # 5) label
    label = float(cfg.arrays[cfg.label][i]) if isinstance(cfg.label, str) else cfg.label
    label = torch.tensor(label, dtype=cfg.dtype)

    # 6) make the DGLGraph
    g = make_graph(node_features, dtype=cfg.dtype)

    # 7) batch_num_nodes and batch_num_edges
    batch_num_nodes = g.batch_num_nodes()
    batch_num_edges = g.batch_num_edges()

    meta = {
        'globals': global_feat,
        'label': label,
        'weight': weight,
        'tracking': tracking,
        'batch_num_nodes': batch_num_nodes,
        'batch_num_edges': batch_num_edges,
    }
    return g, meta

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

def make_graph(node_features: torch.tensor, dtype=torch.float32):

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

    return g