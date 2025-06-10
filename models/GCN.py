import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
file_path = os.getcwd()
sys.path.append(file_path)

import root_gnn_base.dataset as datasets
from root_gnn_base import utils

import gc

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

class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, activation = nn.ReLU, dropout = 0, **kwargs):
        super().__init__()
        print(f'Unused args while creating MLP: {kwargs}')
        self.layers = Make_MLP(in_size, hid_size, hid_size, n_layers-1, activation, dropout)
        self.linear = nn.Linear(hid_size, out_size)
    
    def forward(self, x):
        return self.linear(self.layers(x))

def broadcast_global_to_nodes(g, globals):
    boundaries = g.batch_num_nodes()
    return torch.repeat_interleave(globals, boundaries, dim=0)

def broadcast_global_to_edges(g, globals):
    boundaries = g.batch_num_edges()
    return torch.repeat_interleave(globals, boundaries, dim=0)

def copy_v(edges):
    return {'m_v': edges.dst['h']}

def partial_reset(model : nn.Module):
    in_size = len(model.classify.weight[0])
    out_size = len(model.classify.weight)
    device = next(model.classify.parameters()).device
    torch.manual_seed(2)
    model.classify = nn.Linear(in_size, out_size)
    model.classify.to(device)
    print(model.classify.weight)

def print_model(model: nn.Module):
    print(model)

def print_mlp(layer):
    for l in layer.children():
        if isinstance(l, nn.Linear):
            print(l.state_dict())
        else:
            print(l)


# This function returns a model with the whole GNN completely reset
def full_reset(model : nn.Module):
    mlp_list = [model.node_encoder, model.edge_encoder, model.global_encoder,
                model.node_update, model.edge_update, model.global_update,
                model.global_decoder]
    
    for mlp in mlp_list:
        for layer in mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    partial_reset(model)


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        # two-layer GCN
        self.layers.extend(
            [nn.Linear(in_size, hid_size),] +
            [nn.Linear(hid_size, hid_size) for i in range(n_layers)] +
            [dglnn.GraphConv(hid_size, hid_size) for i in range(n_layers)] +
            [nn.Linear(hid_size, hid_size) for i in range(n_layers)]
        )
        self.classify = nn.Linear(hid_size, out_size)
        #self.dropout = nn.Dropout(0.05)

    def forward(self, g):
        h = g.ndata['features']
        for i, layer in enumerate(self.layers):
            if i >= self.n_layers + 1 and i < self.n_layers * 2 + 1:
                h = layer(g, h)
            else:
                h = layer(h)
            h = F.relu(h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)
    
class GCN_global(nn.Module):
    def __init__(self, in_size, hid_size=4, out_size=1, n_layers=1, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers

        #encoder
        self.node_encoder = Make_MLP(in_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(1, hid_size, hid_size, n_layers, dropout=dropout)

        #GCN
        self.node_update = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.conv = dglnn.GraphConv(hid_size, hid_size)

        #decoder
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def forward(self, g):
        h = self.node_encoder(g.ndata['features'])
        h_global = self.global_encoder(g.batch_num_nodes()[:, None].to(torch.float))
        for i in range(self.n_layers):
            h = self.node_update(h)
            h = self.conv(g, h)
            g.ndata['h'] = h
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)
    
class GCN_global_2way(nn.Module):
    def __init__(self, in_size, hid_size=4, out_size=1, n_layers=1, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers

        #encoder
        self.node_encoder = Make_MLP(in_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(1, hid_size, hid_size, n_layers, dropout=dropout)

        #GCN
        self.node_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.conv = dglnn.GraphConv(hid_size, hid_size)

        #decoder
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def forward(self, g):
        h = self.node_encoder(g.ndata['features'])
        h_global = self.global_encoder(g.batch_num_nodes()[:, None].to(torch.float))
        for i in range(self.n_layers):
            h = self.node_update(torch.cat((h, broadcast_global_to_nodes(g, h_global)), dim = 1))
            h = self.conv(g, h)
            g.ndata['h'] = h
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)

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

class Transferred_Learning(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
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

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})

        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

        self.global_decoder = Make_MLP(pretraining_model['args']['hid_size'], hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def forward(self, g, global_feats):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return self.classify(self.global_decoder(h_global))

class Transferred_Learning_Graph(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, additional_proc_steps=1, dropout=0, **kwargs):
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

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})

        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        self.additional_proc_steps = additional_proc_steps

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

        #GNN
        self.node_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = Make_MLP(4*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)

        #decoder
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x

    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x

    def forward(self, g, global_feats):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        for j in range(self.additional_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))

        h_global = self.global_decoder(h_global)
        return self.classify(h_global)

class Transferred_Learning_Parallel(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

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
        self.classify = nn.Linear(hid_size + pretraining_model['args']['hid_size'], out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return h_global

    def forward(self, g, global_feats):
        pretrained_global = self.Pretrained_Output(g.clone())
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)

        return self.classify(torch.cat((pretrained_global, h_global), dim = 1))
    
class Transferred_Learning_Sequential(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

        #encoder
        self.mlp = Make_MLP(pretraining_model['args']['hid_size'], hid_size, hid_size, n_layers, dropout=dropout)
    
        self.classify = nn.Linear(hid_size, out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return h_global

    def forward(self, g, global_feats):
        pretrained_global = self.Pretrained_Output(g.clone())
        global_features = self.mlp(pretrained_global)
        return self.classify(global_features)


class Transferred_Learning_Message_Passing(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

        #encoder
        self.mlp = Make_MLP(pretraining_model['args']['hid_size']*pretraining_model['args']['n_proc_steps'], hid_size, hid_size, n_layers, dropout=dropout)
    
        self.classify = nn.Linear(hid_size, out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        message_passing = None
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
            if (message_passing is None):
                message_passing = h_global.clone()
            else:
                message_passing = torch.cat((message_passing, h_global.clone()), dim=1)
        h_global = self.TL_global_decoder(h_global)
        return message_passing

    def forward(self, g, global_feats):
        pretrained_global = self.Pretrained_Output(g.clone())
        #print(f"message_passing layers have size = {pretrained_global.shape}")
        #print(pretrained_global)
        global_features = self.mlp(pretrained_global)
        return self.classify(global_features)
    
class Transferred_Learning_Message_Passing_Parallel(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

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

        # Freeze Weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False  # Freeze all layers

        self.classify = nn.Linear(pretraining_model['args']['hid_size']*pretraining_model['args']['n_proc_steps'] + hid_size, out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        message_passing = None
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
            if (message_passing is None):
                message_passing = h_global.clone()
            else:
                message_passing = torch.cat((message_passing, h_global.clone()), dim=1)
        h_global = self.TL_global_decoder(h_global)
        return message_passing

    def forward(self, g, global_feats):
        pretrained_message = self.Pretrained_Output(g.clone())
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(torch.cat((pretrained_message, h_global), dim = 1))
    
class Transferred_Learning_Finetuning(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, frozen_pretraining=False, **kwargs):
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

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})

        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        print(f"Freeze Pretraining = {frozen_pretraining}")
        if (frozen_pretraining):
            for param in self.pretrained_model.parameters():
                param.requires_grad = False  # Freeze all layers
            for param in self.pretrained_model[7]:
                param.requires_grad = True

        torch.manual_seed(2)
        self.classify = nn.Linear(pretraining_model['args']['hid_size'], out_size)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return h_global

    def forward(self, g, global_feats):
        h_global = self.Pretrained_Output(g.clone())
        return self.classify(h_global)
    
    def representation(self, g, global_feats):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        
        before_global_decoder = h_global
        after_global_decoder = self.TL_global_decoder(before_global_decoder)
        after_classify = self.classify(after_global_decoder)
        return before_global_decoder, after_global_decoder, after_classify

    def __str__(self):
        layer_names = ["node_encoder", "edge_encoder", "global_encoder", 
                  "node_update", "edge_update", "global_update", "global_decoder"]
        
        layers = [self.pretrained_model[1], self.pretrained_model[2], self.pretrained_model[3],
                  self.pretrained_model[4], self.pretrained_model[5], self.pretrained_model[6], 
                  self.pretrained_model[7]]
        
        for i in range(len(layers)):
            print(layer_names[i])
            for layer in layers[i].children():
                if isinstance(layer, nn.Linear):
                    print(layer.state_dict())
        
        print("classify")
        print(self.classify.weight)
        return ""
    

class Transferred_Learning_Parallel_Finetuning(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, learning_rate=0.0001, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')

        self.learning_rate = learning_rate

        self.parallel_params = []
        self.finetuning_params = []


        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})
        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        self.finetuning_params.append(self.pretrained_model)

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
        self.classify = nn.Linear(hid_size + pretraining_model['args']['hid_size'], out_size)

        self.parallel_params.append(self.node_encoder)
        self.parallel_params.append(self.edge_encoder)
        self.parallel_params.append(self.global_encoder)
        self.parallel_params.append(self.node_update)
        self.parallel_params.append(self.edge_update)
        self.parallel_params.append(self.global_update)
        self.parallel_params.append(self.global_decoder)
        self.parallel_params.append(self.classify)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_edge_encoder(self, x):
        for layer in self.pretrained_model[2]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_node_update(self, x):
        for layer in self.pretrained_model[4]:
            x = layer(x)
        return x

    def TL_edge_update(self, x):
        for layer in self.pretrained_model[5]:
            x = layer(x)
        return x

    def TL_global_update(self, x):
        for layer in self.pretrained_model[6]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def Pretrained_Output(self, g):
        h = self.TL_node_encoder(g.ndata['features'])
        e = self.TL_edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return h_global

    def forward(self, g, global_feats):
        pretrained_global = self.Pretrained_Output(g.clone())
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)

        return self.classify(torch.cat((pretrained_global, h_global), dim = 1))
    
    def parameters(self, recurse: bool = True):
        params = []
        for model_section in self.parallel_params:
            if (type(self.learning_rate) == dict and self.learning_rate["trainable_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["trainable_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        for model_section in self.finetuning_params:
            if (type(self.learning_rate) == dict and self.learning_rate["finetuning_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["finetuning_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        return params    
    
class Attention(nn.Module):
    def __init__(self, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, num_heads = 1, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        #encoder
        self.node_encoder = Make_MLP(sample_graph.ndata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)

        #GNN
        self.node_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)

        #decoder
        self.global_decoder = Make_MLP(hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)
        
        #attention
        self.multihead_attn = nn.MultiheadAttention(hid_size, num_heads, dropout=dropout, batch_first=True)
        self.queries = nn.Linear(hid_size, hid_size)
        self.keys = nn.Linear(hid_size, hid_size)
        self.values = nn.Linear(hid_size, hid_size)

    def forward(self, g, global_feats):
        h = self.node_encoder(g.ndata['features'])
        g.ndata['h'] = h

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

        h_original_shape = h.shape
        num_graphs = len(dgl.unbatch(g))
        num_nodes = g.batch_num_nodes()[0].item()
        padding_mask = g.ndata['padding_mask'] > 0
        padding_mask = torch.reshape(padding_mask, (num_graphs, num_nodes))

        h = g.ndata['h']
        query = self.queries(h)
        key = self.keys(h)
        value = self.values(h)
        query = torch.reshape(query, (num_graphs, num_nodes, h_original_shape[1]))
        key = torch.reshape(key, (num_graphs, num_nodes, h_original_shape[1]))
        value = torch.reshape(value, (num_graphs, num_nodes, h_original_shape[1]))
        h, _ = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
        h = torch.reshape(h, h_original_shape)
        
        h = self.node_update(torch.cat((h, broadcast_global_to_nodes(g, h_global)), dim = 1))
        g.ndata['h'] = h
        mean_nodes = dgl.sum_nodes(g, 'h', 'w') / sum_weights
        h_global = self.global_update(torch.cat((h_global, mean_nodes), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)

class Attention_Edge_Network(nn.Module):
    def __init__(self, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, num_heads = 1, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
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
        

        #attention
        self.multihead_attn = nn.MultiheadAttention(hid_size, num_heads, dropout=dropout, batch_first=True)
        self.queries = nn.Linear(hid_size, hid_size)
        self.keys = nn.Linear(hid_size, hid_size)
        self.values = nn.Linear(hid_size, hid_size)

    def forward(self, g, global_feats):
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e

        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)

        h = g.ndata['h']
        h_original_shape = h.shape
        num_graphs = len(dgl.unbatch(g))
        num_nodes = g.batch_num_nodes()[0].item()
        padding_mask = g.ndata['padding_mask'] > 0
        padding_mask = torch.reshape(padding_mask, (num_graphs, num_nodes))

        for i in range(self.n_proc_steps):
            
            h = g.ndata['h']
            query = self.queries(h)
            key = self.keys(h)
            value = self.values(h)
            query = torch.reshape(query, (num_graphs, num_nodes, h_original_shape[1]))
            key = torch.reshape(key, (num_graphs, num_nodes, h_original_shape[1]))
            value = torch.reshape(value, (num_graphs, num_nodes, h_original_shape[1]))
            h, _ = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
            h = torch.reshape(h, h_original_shape)

            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h', 'w'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)

class Attention_Unbatched(nn.Module):
    def __init__(self, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, num_heads = 1, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
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
        

        #attention
        self.multihead_attn = nn.MultiheadAttention(hid_size, 1, dropout=dropout)
        self.queries = nn.Linear(hid_size, hid_size)
        self.keys = nn.Linear(hid_size, hid_size)
        self.values = nn.Linear(hid_size, hid_size)



    def forward(self, g, global_feats):

        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e

        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)

        for i in range(self.n_proc_steps):

            unbatched_g = dgl.unbatch(g)
            for graph in unbatched_g:
                h = graph.ndata['h']
                h, _ = self.multihead_attn(self.queries(h), self.keys(h), self.values(h))
                graph.ndata['h'] = h
            g = dgl.batch(unbatched_g)

            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = self.global_decoder(h_global)
        return self.classify(h_global)
    
class Transferred_Learning_Attention(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, num_heads, dropout=0, learning_rate=0.0001, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.learning_rate = learning_rate

        self.pretraining_params = []
        self.attention_params = []
        
        self.pretrained_model = utils.buildFromConfig(pretraining_model, {'sample_graph': sample_graph, 'sample_global': sample_global})

        checkpoint = torch.load(pretraining_path)
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_layers = list(self.pretrained_model.children())
        pretrained_layers = pretrained_layers[:-1]
        self.pretrained_model = nn.Sequential(*pretrained_layers)

        self.pretraining_params.append(self.pretrained_model[1])
        self.pretraining_params.append(self.pretrained_model[3])
        self.pretraining_params.append(self.pretrained_model[7])

        #attention
        self.multihead_attn = nn.MultiheadAttention(hid_size, num_heads, dropout=dropout, batch_first=True)
        self.queries = nn.Linear(hid_size, hid_size)
        self.keys = nn.Linear(hid_size, hid_size)
        self.values = nn.Linear(hid_size, hid_size)

        self.node_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(2*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        
        self.classify = nn.Linear(pretraining_model['args']['hid_size'], out_size)

        self.attention_params.append(self.multihead_attn)

        self.attention_params.append(self.queries)
        self.attention_params.append(self.keys)
        self.attention_params.append(self.values)
        self.attention_params.append(self.classify)
        self.attention_params.append(self.node_update)
        self.attention_params.append(self.global_update)

    def TL_node_encoder(self, x):
        for layer in self.pretrained_model[1]:
            x = layer(x)
        return x
    
    def TL_global_encoder(self, x):
        for layer in self.pretrained_model[3]:
            x = layer(x)
        return x
    
    def TL_global_decoder(self, x):
        for layer in self.pretrained_model[7]:
            x = layer(x)
        return x

    def forward(self, g, global_feats):
        h = self.TL_node_encoder(g.ndata['features'])
        g.ndata['h'] = h

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

        h_global = self.TL_global_encoder(global_feats)

        h_original_shape = h.shape
        num_graphs = len(dgl.unbatch(g))
        num_nodes = g.batch_num_nodes()[0].item()
        padding_mask = g.ndata['padding_mask'] > 0
        padding_mask = torch.reshape(padding_mask, (num_graphs, num_nodes))

        h = g.ndata['h']
        query = self.queries(h)
        key = self.keys(h)
        value = self.values(h)
        query = torch.reshape(query, (num_graphs, num_nodes, h_original_shape[1]))
        key = torch.reshape(key, (num_graphs, num_nodes, h_original_shape[1]))
        value = torch.reshape(value, (num_graphs, num_nodes, h_original_shape[1]))
        h, _ = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
        h = torch.reshape(h, h_original_shape)
        
        h = self.node_update(torch.cat((h, broadcast_global_to_nodes(g, h_global)), dim = 1))
        g.ndata['h'] = h
        mean_nodes = dgl.sum_nodes(g, 'h', 'w') / sum_weights
        h_global = self.global_update(torch.cat((h_global, mean_nodes), dim = 1))
        h_global = self.TL_global_decoder(h_global)
        return self.classify(h_global)
    
    def parameters(self, recurse: bool = True):
        params = []
        for model_section in self.pretraining_params:
            if (type(self.learning_rate) == dict and self.learning_rate["pretraining_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["pretraining_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        for model_section in self.attention_params:
            if (type(self.learning_rate) == dict and self.learning_rate["attention_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["attention_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        return params
    
class Multimodel_Transferred_Learning(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, frozen_pretraining=True, learning_rate=None, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.learning_rate = learning_rate
        input_size = 0

        self.pretraining_params = []
        self.model_params = []
        
        self.pretrained_models = []
        for model, path in zip(pretraining_model, pretraining_path):
            input_size += model['args']['hid_size']
            model = utils.buildFromConfig(model, {'sample_graph': sample_graph, 'sample_global': sample_global})

            checkpoint = torch.load(path)['model_state_dict']
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            pretrained_layers = list(model.children())
            pretrained_layers = pretrained_layers[:-1]
        
            model = nn.Sequential(*pretrained_layers)

            # Freeze Weights
            print(f"Freeze Pretraining = {frozen_pretraining}")
            if (frozen_pretraining):
                for param in model.parameters():
                    param.requires_grad = False  # Freeze all layers
            self.pretraining_params.append(model)
            self.pretrained_models.append(model)

        print(f"len(pretrained_models) = {len(self.pretrained_models)}")
        print(f"input size = {input_size}")

        self.final_mlp = Make_MLP(input_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

        self.model_params.append(self.final_mlp)
        self.model_params.append(self.classify)

    def TL_node_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][1]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][1]:
                x = layer(x)
            return x
    
    def TL_edge_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][2]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][2]:
                x = layer(x)
            return x
    
    def TL_global_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][3]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][3]:
                x = layer(x)
            return x
    
    def TL_node_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][4]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][4]:
                x = layer(x)
            return x

    def TL_edge_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][5]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][5]:
                x = layer(x)
            return x

    def TL_global_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][6]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][6]:
                x = layer(x)
            return x
    
    def TL_global_decoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][7]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][7]:
                x = layer(x)
            return x

    def Pretrained_Output(self, g, model_idx):
        h = self.TL_node_encoder(g.ndata['features'], model_idx)
        e = self.TL_edge_encoder(g.edata['features'], model_idx)
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats, model_idx)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1), model_idx)
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1), model_idx)
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1), model_idx)
        # h_global = self.TL_global_decoder(h_global, model_idx)
        return h_global

    def forward(self, g, global_feats):
        h_global = []
        for i in range(len(self.pretrained_models)):
            h_global.append(self.Pretrained_Output(g.clone(), i))
        h_global = torch.concatenate(h_global, dim=1)
        return self.classify(self.final_mlp(h_global))
    
    def to(self, device):
        for i in range(len(self.pretrained_models)):
            self.pretrained_models[i].to(device)
        self.classify.to(device)
        self.final_mlp.to(device)
        return self
    
    def parameters(self, recurse: bool = True):
        params = []
        for model_section in self.pretraining_params:
            if (type(self.learning_rate) == dict and self.learning_rate["pretraining_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["pretraining_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.00001})
        for model_section in self.model_params:
            if (type(self.learning_rate) == dict and self.learning_rate["model_lr"]): 
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["model_lr"]})
            else:
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        return params    


class MultiModel(nn.Module):
    def __init__(self, pretraining_path, pretraining_model, sample_graph, sample_global, hid_size, out_size, n_layers, n_proc_steps, dropout=0, frozen_pretraining=True, learning_rate=None, **kwargs):
        super().__init__()
        print(f'Unused args while creating GCN: {kwargs}')
        self.n_layers = n_layers
        self.n_proc_steps = n_proc_steps
        self.layers = nn.ModuleList()
        self.has_global = sample_global.shape[1] != 0
        gl_size = sample_global.shape[1] if self.has_global else 1

        self.learning_rate = learning_rate
        input_size = 0

        self.model_params = []
        self.pretraining_params = []
        
        self.pretrained_models = []
        for model, path in zip(pretraining_model, pretraining_path):
            input_size += model['args']['hid_size']
            model = utils.buildFromConfig(model, {'sample_graph': sample_graph, 'sample_global': sample_global})

            checkpoint = torch.load(path)['model_state_dict']
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            pretrained_layers = list(model.children())
            pretrained_layers = pretrained_layers[:-1]
        
            model = nn.Sequential(*pretrained_layers)

            # Freeze Weights
            print(f"Freeze Pretraining = {frozen_pretraining}")
            if (frozen_pretraining):
                for param in model.parameters():
                    param.requires_grad = False  # Freeze all layers
            self.pretraining_params.append(model)
            self.pretrained_models.append(model)

        print(f"len(pretrained_models) = {len(self.pretrained_models)}")
        print(f"input size = {input_size}")

        #encoder
        self.node_encoder = Make_MLP(sample_graph.ndata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_encoder = Make_MLP(sample_graph.edata['features'].shape[1], hid_size, hid_size, n_layers, dropout=dropout)
        self.global_encoder = Make_MLP(gl_size, hid_size, hid_size, n_layers, dropout=dropout)

        #GNN
        self.node_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.edge_update = Make_MLP(4*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.global_update = Make_MLP(3*hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        
        self.final_mlp = Make_MLP(input_size + hid_size, hid_size, hid_size, n_layers, dropout=dropout)
        self.classify = nn.Linear(hid_size, out_size)

        self.model_params.append(self.final_mlp)
        self.model_params.append(self.classify)

    def TL_node_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][1]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][1]:
                x = layer(x)
            return x
    
    def TL_edge_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][2]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][2]:
                x = layer(x)
            return x
    
    def TL_global_encoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][3]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][3]:
                x = layer(x)
            return x
    
    def TL_node_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][4]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][4]:
                x = layer(x)
            return x

    def TL_edge_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][5]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][5]:
                x = layer(x)
            return x

    def TL_global_update(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][6]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][6]:
                x = layer(x)
            return x
    
    def TL_global_decoder(self, x, model_idx):
        try:
            for layer in self.pretrained_models[model_idx][7]:
                x = layer(x)
            return x
        except (NotImplementedError, IndexError):
            for layer in self.pretrained_models[model_idx][1][7]:
                x = layer(x)
            return x

    def Pretrained_Output(self, g, model_idx):
        h = self.TL_node_encoder(g.ndata['features'], model_idx)
        e = self.TL_edge_encoder(g.edata['features'], model_idx)
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.TL_global_encoder(global_feats, model_idx)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.TL_edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1), model_idx)
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.TL_node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1), model_idx)
            h_global = self.TL_global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1), model_idx)
        # h_global = self.TL_global_decoder(h_global, model_idx)
        return h_global

    def forward(self, g, global_feats):
        h = self.node_encoder(g.ndata['features'])
        e = self.edge_encoder(g.edata['features'])
        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)
        h_global = self.global_encoder(global_feats)
        for i in range(self.n_proc_steps):
            g.apply_edges(dgl.function.copy_u('h', 'm_u'))
            g.apply_edges(copy_v)
            g.edata['e'] = self.edge_update(torch.cat((g.edata['e'], g.edata['m_u'], g.edata['m_v'], broadcast_global_to_edges(g, h_global)), dim = 1))
            g.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'h_e'))
            g.ndata['h'] = self.node_update(torch.cat((g.ndata['h'], g.ndata['h_e'], broadcast_global_to_nodes(g, h_global)), dim = 1))
            h_global = self.global_update(torch.cat((h_global, dgl.mean_nodes(g, 'h'), dgl.mean_edges(g, 'e')), dim = 1))
        h_global = [h_global]
        for i in range(len(self.pretrained_models)):
            h_global.append(self.Pretrained_Output(g.clone(), i))
        h_global = torch.concatenate(h_global, dim=1)
        return self.classify(self.final_mlp(h_global))
    
    def to(self, device):
        for i in range(len(self.pretrained_models)):
            self.pretrained_models[i].to(device)
        self.classify.to(device)
        self.final_mlp.to(device)
        self.node_encoder.to(device)
        self.edge_encoder.to(device)
        self.global_encoder.to(device)

        self.node_update.to(device)
        self.edge_update.to(device)
        self.global_update.to(device)
        return self
    
    def parameters(self, recurse: bool = True):
        params = []
        for i, model_section in enumerate(self.pretraining_params):
            if (type(self.learning_rate) == dict and self.learning_rate["pretraining_lr"]): 
                print(f"Pretraining LR = {self.learning_rate['pretraining_lr'][i]}")
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["pretraining_lr"][i]})
            else:
                print(f"Pretraining LR = 0.00001")
                params.append({'params': model_section.parameters(), 'lr': 0.00001})
        for model_section in self.model_params:
            if (type(self.learning_rate) == dict and self.learning_rate["model_lr"]): 
                print(f"Model LR = {self.learning_rate['model_lr']}")
                params.append({'params': model_section.parameters(), 'lr': self.learning_rate["model_lr"]})
            else:
                print(f"Model LR = 0.0001")
                params.append({'params': model_section.parameters(), 'lr': 0.0001})
        return params    


class Clustering(nn.Module):
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
        self.global_decoder = Make_MLP(hid_size, hid_size, out_size, n_layers, dropout=dropout)

    def model_forward(self, g, global_feats, features = 'features'):
        h = self.node_encoder(g.ndata[features])
        e = self.edge_encoder(g.edata[features])

        g.ndata['h'] = h
        g.edata['e'] = e
        if not self.has_global:
            global_feats = g.batch_num_nodes()[:, None].to(torch.float)

        batch_num_nodes = None
        sum_weights = None
        if "w" in g.ndata:
            batch_indices = g.batch_num_nodes()
            # Find non-zero rows (non-padded nodes)
            non_padded_nodes_mask = torch.any(g.ndata[features] != 0, dim=1)
            # Split the mask according to the batch indices
            batch_num_nodes = []
            start_idx = 0
            for num_nodes in batch_indices:
                end_idx = start_idx + num_nodes
                non_padded_count = non_padded_nodes_mask[start_idx:end_idx].sum().item()
                batch_num_nodes.append(non_padded_count)
                start_idx = end_idx
            batch_num_nodes = torch.tensor(batch_num_nodes, device = g.ndata[features].device)
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
        return h_global
    
    def forward(self, g, global_feats):
        h_global = self.model_forward(g, global_feats, 'features')
        h_global_augmented = self.model_forward(g, global_feats, 'augmented_features')
        return torch.cat((h_global, h_global_augmented), dim=1)

    def representation(self, g, global_feats):
        h_global = self.model_forward(g, global_feats, 'features')
        h_global_augmented = self.model_forward(g, global_feats, 'augmented_features')
        return h_global, h_global_augmented, torch.cat((h_global, h_global_augmented), dim=1)
    
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