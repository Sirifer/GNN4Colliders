import torch
import torch.nn as nn
import dgl

def mean_nodes(batched_graph, feat_key='h', op='mean', node_split=None):
    """
    Aggregates node features per disjoint graph in a batched DGLGraph.

    Args:
        batched_graph: DGLGraph
        feat_key: str, node feature key
        op: 'mean', 'sum', or 'max'
        node_split: 1D tensor or list of ints (num nodes per graph)

    Returns:
        Tensor of shape [num_graphs, node_feat_dim]
    """
    h = batched_graph.ndata[feat_key]
    if node_split is None or len(node_split) == 0:
        if op == 'mean':
            return dgl.mean_nodes(batched_graph, feat_key)
        elif op == 'sum':
            return dgl.sum_nodes(batched_graph, feat_key)
        elif op == 'max':
            return dgl.max_nodes(batched_graph, feat_key)
        else:
            raise ValueError(f"Unknown op: {op}")
    else:
        # Ensure node_split is a flat list of ints
        if isinstance(node_split, torch.Tensor):
            splits = node_split.view(-1).tolist()
        else:
            splits = [int(x) for x in node_split]
        chunks = torch.split(h, splits, dim=0)
        if op == 'mean':
            out = torch.stack([chunk.mean(0) if chunk.shape[0] > 0 else torch.zeros_like(h[0]) for chunk in chunks])
        elif op == 'sum':
            out = torch.stack([chunk.sum(0) if chunk.shape[0] > 0 else torch.zeros_like(h[0]) for chunk in chunks])
        elif op == 'max':
            out = torch.stack([chunk.max(0).values if chunk.shape[0] > 0 else torch.zeros_like(h[0]) for chunk in chunks])
        else:
            raise ValueError(f"Unknown op: {op}")
        return out
    
def mean_edges(batched_graph, feat_key='e', op='mean', edge_split=None):
    """
    Aggregates edge features per disjoint graph in a batched DGLGraph.

    Args:
        batched_graph: DGLGraph
        feat_key: str, edge feature key
        op: 'mean', 'sum', or 'max'
        edge_split: 1D tensor or list of ints (num edges per graph)

    Returns:
        Tensor of shape [num_graphs, edge_feat_dim]
    """
    e = batched_graph.edata[feat_key]
    if edge_split is None or len(edge_split) == 0:
        if op == 'mean':
            return dgl.mean_edges(batched_graph, feat_key)
        elif op == 'sum':
            return dgl.sum_edges(batched_graph, feat_key)
        elif op == 'max':
            return dgl.max_edges(batched_graph, feat_key)
        else:
            raise ValueError(f"Unknown op: {op}")
    else:
        # Ensure edge_split is a flat list of ints
        if isinstance(edge_split, torch.Tensor):
            splits = edge_split.view(-1).tolist()
        else:
            splits = [int(x) for x in edge_split]
        chunks = torch.split(e, splits, dim=0)
        if op == 'mean':
            out = torch.stack([chunk.mean(0) if chunk.shape[0] > 0 else torch.zeros_like(e[0]) for chunk in chunks])
        elif op == 'sum':
            out = torch.stack([chunk.sum(0) if chunk.shape[0] > 0 else torch.zeros_like(e[0]) for chunk in chunks])
        elif op == 'max':
            out = torch.stack([chunk.max(0).values if chunk.shape[0] > 0 else torch.zeros_like(e[0]) for chunk in chunks])
        else:
            raise ValueError(f"Unknown op: {op}")
        return out
    
def Make_SLP(in_size, out_size, activation = nn.ReLU, dropout = 0):
    layers = []
    layers.append(nn.Linear(in_size, out_size))
    layers.append(activation())
    layers.append(nn.Dropout(dropout))
    return layers

def Make_MLP(in_size, hid_size, out_size, n_layers, activation = nn.ReLU, dropout = 0):
    layers = []
    if n_layers > 1:
        layers += Make_SLP(in_size, hid_size, activation, dropout)
        for i in range(n_layers-2):
            layers += Make_SLP(hid_size, hid_size, activation, dropout)
        layers += Make_SLP(hid_size, out_size, activation, dropout)
    else:
        layers += Make_SLP(in_size, out_size, activation, dropout)
    layers.append(torch.nn.LayerNorm(out_size))
    return nn.Sequential(*layers)

def broadcast_global_to_nodes(globals, node_split):
    """
    globals: [num_graphs, global_dim]
    node_split: list/1D tensor of length num_graphs, number of nodes per graph
    Returns: [total_num_nodes, global_dim]
    """
    if node_split is None:
        raise ValueError("node_split must be provided")
    if not torch.is_tensor(node_split):
        node_split = torch.tensor(node_split, dtype=torch.long, device=globals.device)
    else:
        node_split = node_split.to(device=globals.device, dtype=torch.long)
    node_split = node_split.flatten()
    return torch.repeat_interleave(globals, node_split, dim=0)

def broadcast_global_to_edges(globals, edge_split):
    """
    globals: [num_graphs, global_dim] (on CUDA or CPU)
    edge_split: list/1D tensor of length num_graphs, number of edges per graph (CPU or CUDA)
    Returns: [total_num_edges, global_dim]
    """
    if edge_split is None:
        raise ValueError("edge_split must be provided")
    if not torch.is_tensor(edge_split):
        edge_split = torch.tensor(edge_split, dtype=torch.long, device=globals.device)
    else:
        edge_split = edge_split.to(device=globals.device, dtype=torch.long)
    edge_split = edge_split.flatten()
    return torch.repeat_interleave(globals, edge_split, dim=0)

def copy_v(edges):
    return {'m_v': edges.dst['h']}