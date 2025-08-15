import torch
import torch.nn as nn
import dgl

from models import utils

class Edge_Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid_size = cfg.hid_size
        n_layers = cfg.n_layers
        self.n_proc_steps = cfg.n_proc_steps

        #encoder
        self.node_encoder = utils.Make_MLP(cfg.input_dim_nodes, hid_size, hid_size, n_layers)
        self.edge_encoder = utils.Make_MLP(cfg.input_dim_edges, hid_size, hid_size, n_layers)
        self.global_encoder = utils.Make_MLP(cfg.input_dim_globals, hid_size, hid_size, n_layers)

        #GNN
        self.node_update = utils.Make_MLP(3*hid_size, hid_size, hid_size, n_layers)
        self.edge_update = utils.Make_MLP(4*hid_size, hid_size, hid_size, n_layers)
        self.global_update = utils.Make_MLP(3*hid_size, hid_size, hid_size, n_layers)

        #decoder
        self.global_decoder = utils.Make_MLP(hid_size, hid_size, hid_size, n_layers)
        self.classify = nn.Linear(hid_size, cfg.out_dim)

    def forward(self, node_feats, edge_feats, global_feats, batched_graph, metadata={}):
        # encoders
        batched_graph.ndata['h'] = self.node_encoder(node_feats)
        batched_graph.edata['e'] = self.edge_encoder(edge_feats)

        if global_feats.ndim == 3:
            global_feats = global_feats.view(-1, global_feats.shape[-1])
        h_global = self.global_encoder(global_feats)

        # message passing
        for _ in range(self.n_proc_steps):
            batched_graph.apply_edges(dgl.function.copy_u('h', 'm_u'))
            batched_graph.apply_edges(utils.copy_v)

            # edge update
            edge_inputs = torch.cat([
                batched_graph.edata['e'],
                batched_graph.edata['m_u'],
                batched_graph.edata['m_v'],
                utils.broadcast_global_to_edges(h_global, edge_split=metadata.get("batch_num_edges", None))
            ], dim=1)
            batched_graph.edata['e'] = self.edge_update(edge_inputs)

            # node update
            batched_graph.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            node_inputs = torch.cat([
                batched_graph.ndata['h'],
                batched_graph.ndata['h_e'],
                utils.broadcast_global_to_nodes(h_global, node_split=metadata.get("batch_num_nodes", None))
            ], dim=1)
            batched_graph.ndata['h'] = self.node_update(node_inputs)

            # global update
            graph_node_feat = utils.mean_nodes(
                batched_graph, 'h', node_split=metadata.get("batch_num_nodes", None)
            )
            graph_edge_feat = utils.mean_edges(
                batched_graph, 'e', edge_split=metadata.get("batch_num_edges", None)
            )
            h_global = self.global_update(torch.cat([h_global, graph_node_feat, graph_edge_feat], dim=1))

        h_global = self.global_decoder(h_global)
        out = self.classify(h_global)
        return out

