import torch
import torch.nn as nn
import dgl

from models import utils

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

        graph_node_feat = utils.mean_nodes(batched_graph, 'h', node_split=metadata.get("batch_num_nodes", None))
        graph_edge_feat = utils.mean_edges(batched_graph, 'e', edge_split=metadata.get("batch_num_edges", None))

        # Flatten global_feats if needed
        if global_feats.ndim == 3:
            global_feats = global_feats.view(-1, global_feats.shape[-1])
        global_emb = self.global_mlp(global_feats)  # [num_graphs, global_emb_dim]

        combined_feat = torch.cat([graph_node_feat, graph_edge_feat, global_emb], dim=-1)
        graph_pred = self.mlp(combined_feat)
        return graph_pred
    
