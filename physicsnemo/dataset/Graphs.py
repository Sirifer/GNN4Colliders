import dgl
import torch
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Graphs:
    graphs: List[dgl.DGLGraph]
    metadata: Dict[str, torch.Tensor]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        meta = {k: v[idx] for k, v in self.metadata.items()}
        return self.graphs[idx], meta

    def shuffle(self):
        idx = torch.randperm(len(self.graphs))
        self.graphs = [self.graphs[i] for i in idx]
        for k in self.metadata:
            self.metadata[k] = self.metadata[k][idx]

    def batch(self, batch_size, node_feature_dim=None, dtype=None):
        """
        In-place batching: after this, self.graphs is a list of batched DGLGraphs,
        and self.metadata[k] is a tensor of shape [num_batches, batch_size, ...].
        """
        batched_graphs = []
        batched_meta = {k: [] for k in self.metadata}
        N = len(self.graphs)

        # Infer node_feature_dim and dtype if not specified
        if node_feature_dim is None and N > 0:
            feats = self.graphs[0].ndata['features']
            node_feature_dim = feats.shape[1] if feats.ndim > 1 else 1
        if dtype is None and N > 0:
            dtype = self.graphs[0].ndata['features'].dtype

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_graphs = self.graphs[start:end]
            batch_meta = {k: v[start:end] for k, v in self.metadata.items()}

            # Padding if needed
            pad_count = batch_size - len(batch_graphs)
            if pad_count > 0:
                dummy_graph = dgl.graph(([], []))
                dummy_graph.ndata['features'] = torch.empty((0, node_feature_dim), dtype=dtype)
                dummy_graph.edata['features'] = torch.empty((0, 3), dtype=dtype)  # assuming 3 edge features
                batch_graphs += [dummy_graph] * pad_count

                # Pad metadata with zeros
                for k, v in batch_meta.items():
                    shape = list(v[0].shape) if len(v) > 0 else []
                    pad_tensor = torch.zeros([pad_count] + shape, dtype=v.dtype, device=v.device)
                    batch_meta[k] = torch.cat([v, pad_tensor], dim=0)
            else:
                for k, v in batch_meta.items():
                    batch_meta[k] = torch.stack(v, dim=0) if isinstance(v, list) else v

            batched_graphs.append(dgl.batch(batch_graphs))
            for k in batched_meta:
                batched_meta[k].append(batch_meta[k])

        # Now stack along a new axis: [num_batches, batch_size, ...]
        for k in batched_meta:
            self.metadata[k] = torch.stack(batched_meta[k], dim=0)

        self.graphs = batched_graphs

    def normalize(self, stats):
        node_mean, node_std, _ = stats['node']
        edge_mean, edge_std, _ = stats['edge']
        for g in self.graphs:
            g.ndata['features'] = (g.ndata['features'] - node_mean) / node_std
            g.edata['features'] = (g.edata['features'] - edge_mean) / edge_std

def save_graphs(graphs: Graphs, f: str):
    meta_to_save = {k: v for k, v in graphs.metadata.items()}
    dgl.save_graphs(f, graphs.graphs, meta_to_save)

def load_graphs(f: str) -> Graphs:
    g, meta = dgl.load_graphs(f)
    for k in meta:
        if not isinstance(meta[k], torch.Tensor):
            meta[k] = torch.stack(meta[k])
    return Graphs(graphs=g, metadata=meta)