import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
file_path = os.getcwd()
sys.path.append(file_path)

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

class Edge_Network(nn.Module):
    def __init__(self, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        if (len(sample_global) == 0):
            self.has_global = False
        else:
            self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        #encoder
        self.node_encoder = Make_MLP(sample_graph.ndata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_encoder = Make_MLP(sample_graph.edata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)

        #GNN
        self.node_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = Make_MLP(4*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)

        #decoder
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def forward(self, g, global_feats):
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])

        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)

        batch_num_nodes = None
        sum_weights = None
        if "w" in g.ndata:
            batch_indices = g.batch_num_nodes()
            # Find non-zero rows (non-padded nodes)
            non_padded_nodes_mask = torch.any(g.ndata['features'] != 0, dim=1)
            # Split the mask according to the batch indices
            batch_num_nodes = []
            start_idx = 0
            for num_nodes in batch_indices:
                end_idx = start_idx + num_nodes
                non_padded_count = non_padded_nodes_mask[start_idx:end_idx].sum().item()
                batch_num_nodes.append(non_padded_count)
                start_idx = end_idx
            batch_num_nodes = torch.tensor(batch_num_nodes, device = g.ndata['features'].device)
            sum_weights = batch_num_nodes[:, None].repeat(1, 64)
            global_feats = batch_num_nodes[:, None].to(torch.float)

        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            if "w" in g.ndata:
                mean_nodes = dgl.sum_nodes(g, 'h', 'w') / sum_weights
                h_global = self.global_update(torch.cat((h_global, mean_nodes, dgl.mean_edges(g, 'e')), dim = 1))
            else:
                h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)
    
    def representation(self, g, global_feats):
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])

        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)

        batch_num_nodes = None
        sum_weights = None
        if "w" in g.ndata:
            batch_indices = g.batch_num_nodes()
            # Find non-zero rows (non-padded nodes)
            non_padded_nodes_mask = torch.any(g.ndata['features'] != 0, dim=1)
            # Split the mask according to the batch indices
            batch_num_nodes = []
            start_idx = 0
            for num_nodes in batch_indices:
                end_idx = start_idx + num_nodes
                non_padded_count = non_padded_nodes_mask[start_idx:end_idx].sum().item()
                batch_num_nodes.append(non_padded_count)
                start_idx = end_idx
            batch_num_nodes = torch.tensor(batch_num_nodes, device = g.ndata['features'].device)
            sum_weights = batch_num_nodes[:, None].repeat(1, 64)
            global_feats = batch_num_nodes[:, None].to(torch.float)

        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            if "w" in g.ndata:
                mean_nodes = dgl.sum_nodes(g, 'h', 'w') / sum_weights
                h_global = self.global_update(torch.cat((h_global, mean_nodes, dgl.mean_edges(g, 'e')), dim = 1))
            else:
                h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        before_global_decoder = h_global
        after_global_decoder = self.global_decoder(before_global_decoder)
        after_classify = self.classify(after_global_decoder)
        return before_global_decoder, after_global_decoder, after_classify
    
    def __str__(self):
        layer_names = ["node_encoder", "edge_encoder", "global_encoder", 
                  "node_update", "edge_update", "global_update", "global_decoder"]
        
        layers = [self.node_encoder, self.edge_encoder, self.global_encoder,
                  self.node_update, self.edge_update, self.global_update, self.global_decoder]
        
        for i in range(len(layers)):
            print(layer_names[i])
            for layer in layers[i].children():
                if isinstance(layer, nn.Linear):
                    print(layer.state_dict())
        
        print("classify")
        print(self.classify.weight)
        return ""

    def __name__():
        return "Edge_Network"