import torch
import torch.nn as nn
import dgl

# Import the PhysicsNemo MeshGraphNet model
from physicsnemo.models.meshgraphnet import MeshGraphNet as PhysicsNemoMeshGraphNet

class MeshGraphNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base_gnn_cfg = cfg.base_gnn
        self.base_gnn = PhysicsNemoMeshGraphNet(**base_gnn_cfg)

        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.global_feat_dim, cfg.global_emb_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Linear(
            base_gnn_cfg['output_dim'] + base_gnn_cfg['input_dim_edges'] + cfg.global_emb_dim,
            cfg.out_dim
        )
    
    def forward(self, node_feats, edge_feats, global_feats, batched_graph, metadata={}):
        """
        node_feats: [total_num_nodes, node_feat_dim]
        edge_feats: [total_num_edges, edge_feat_dim]
        global_feats: [num_graphs, global_feat_dim]
        batched_graph: DGLGraph, representing the collection of graphs in a batch
        metadata: dict, may contain 'batch_num_nodes', 'batch_num_edges', etc.
        Returns:
            graph_pred: [num_graphs, out_dim]
        """
        node_pred = self.base_gnn(node_feats, edge_feats, batched_graph)
        batched_graph.ndata['h'] = node_pred
        batched_graph.edata['e'] = edge_feats

        graph_node_feat = mean_nodes(batched_graph, 'h', node_split=metadata.get("batch_num_nodes", None))
        graph_edge_feat = mean_edges(batched_graph, 'e', edge_split=metadata.get("batch_num_edges", None))

        # Flatten global_feats if needed
        if global_feats.ndim == 3:
            global_feats = global_feats.view(-1, global_feats.shape[-1])
        global_emb = self.global_mlp(global_feats)  # [num_graphs, global_emb_dim]

        combined_feat = torch.cat([graph_node_feat, graph_edge_feat, global_emb], dim=-1)
        graph_pred = self.mlp(combined_feat)
        return graph_pred
    
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