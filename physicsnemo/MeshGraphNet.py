import torch
import torch.nn as nn
import dgl

# Import the PhysicsNemo MeshGraphNet model
from physicsnemo.models.meshgraphnet import MeshGraphNet as PhysicsNemoMeshGraphNet

class MeshGraphNet(nn.Module):
    def __init__(self, *args, out_dim=1, **kwargs):
        super().__init__()
        # Initialize the PhysicsNemo MeshGraphNet
        self.base_gnn = PhysicsNemoMeshGraphNet(*args, **kwargs)
        # Assume node_output_dim is known or infer from args/kwargs
        node_output_dim = kwargs.get('hidden_dim_node_decoder', 64)
        self.mlp = nn.Linear(node_output_dim, out_dim)

    def forward(self, node_feats, edge_feats, batched_graph):
        """
        Args:
            node_feats: [total_num_nodes, node_feat_dim]
            edge_feats: [total_num_edges, edge_feat_dim]
            batched_graph: DGLGraph, batched graphs
        Returns:
            graph_pred: [num_graphs, out_dim]
        """
        node_pred = self.base_gnn(node_feats, edge_feats, batched_graph)
        batched_graph.ndata['h'] = node_pred
        graph_feat = dgl.readout_nodes(batched_graph, 'h', op='mean')  # [num_graphs, node_output_dim]
        graph_pred = self.mlp(graph_feat)  # [num_graphs, out_dim]
        return graph_pred