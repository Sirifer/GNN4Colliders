import torch
import json
import os
from dataset.Graphs import Graphs
from typing import List, Dict, Tuple

def combine_feature_stats(chunks: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Combine mean/std/count from multiple chunks using Welford's algorithm.
    Returns combined mean, std, and total count.
    """
    n_total = 0
    mean_total = None
    M2_total = None

    for chunk in chunks:
        n_k = chunk['count']
        if n_k == 0:
            continue

        mean_k = torch.tensor(chunk['mean'])
        std_k = torch.tensor(chunk['std'])
        M2_k = (std_k ** 2) * n_k

        if n_total == 0:
            mean_total = mean_k
            M2_total = M2_k
            n_total = n_k
        else:
            delta = mean_k - mean_total
            N = n_total + n_k
            mean_total += delta * (n_k / N)
            M2_total += M2_k + (delta ** 2) * (n_total * n_k / N)
            n_total = N

    if n_total == 0:
        return torch.tensor([]), torch.tensor([]), 0

    std_total = torch.sqrt(M2_total / n_total)
    return mean_total, std_total, n_total

def global_stats(dirpath: str, dtype: torch.dtype) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Load all JSON stats files in a directory, combine node, edge, and global stats,
    and optionally save the combined stats as JSON to `save_path`.
    """

    combined_stats_path = os.path.join(dirpath, "global_stats.json")

    if not os.path.exists(combined_stats_path):
        stats_list = []
        for fname in os.listdir(dirpath):
            if fname.endswith('.json'):
                with open(os.path.join(dirpath, fname), 'r') as f:
                    stats_list.append(json.load(f))

        node_stats = [s['node'] for s in stats_list]
        edge_stats = [s['edge'] for s in stats_list]

        combined = {
            'node': combine_feature_stats(node_stats),
            'edge': combine_feature_stats(edge_stats),
        }

        combined_json = {}
        for key, (mean, std, count) in combined.items():
            combined_json[key] = {
                'mean': mean.tolist() if mean.numel() > 0 else [],
                'std': std.tolist() if std.numel() > 0 else [],
                'count': count,
            }

        with open(combined_stats_path, 'w') as f:
            json.dump(combined_json, f, indent=4)

    with open(combined_stats_path, 'r') as f:
        combined_json = json.load(f)

    def to_tensor(d):
        mean = torch.tensor(d['mean'], dtype=dtype) if d['mean'] else torch.tensor([], dtype=dtype)
        std = torch.tensor(d['std'], dtype=dtype) if d['std'] else torch.tensor([], dtype=dtype)
        count = d['count']
        return mean, std, count

    return {
        'node': to_tensor(combined_json['node']),
        'edge': to_tensor(combined_json['edge']),
    }

def compute_stats(feats, eps=1e-6):
    mean = feats.mean(dim=0)
    if feats.size(0) > 1:
        var = ((feats - mean) ** 2).mean(dim=0)
    else:
        var = torch.zeros_like(mean)
    std = torch.sqrt(var)
    std = torch.where(std < eps, torch.full_like(std, eps), std)

    return mean, std

def save_stats(graphs: 'Graphs', filepath: str, categorical_unique_threshold=50):
    """
    Compute and save normalization stats (mean, std, counts) for node and edge features.
    Categorical features (few unique values) have normalization disabled (mean=0, std=1).
    """
    if len(graphs) == 0:
        raise ValueError("No graphs to compute stats from.")

    # Node and edge features
    all_node_feats = torch.cat([g.ndata['features'] for g, _ in graphs], dim=0)
    all_edge_feats = torch.cat([g.edata['features'] for g, _ in graphs], dim=0)

    counts = {
        'node': all_node_feats.size(0),
        'edge': all_edge_feats.size(0),
    }

    node_mean, node_std = compute_stats(all_node_feats)
    edge_mean, edge_std = compute_stats(all_edge_feats)

    categorical_mask = torch.tensor([
        torch.unique(all_node_feats[:, i]).numel() < categorical_unique_threshold
        for i in range(node_mean.size(0))
    ], dtype=torch.bool)
    node_mean[categorical_mask] = 0.0
    node_std[categorical_mask] = 1.0

    stats = {
        'node': {
            'mean': node_mean.tolist(),
            'std': node_std.tolist(),
            'count': counts['node'],
        },
        'edge': {
            'mean': edge_mean.tolist(),
            'std': edge_std.tolist(),
            'count': counts['edge'],
        },
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=4)