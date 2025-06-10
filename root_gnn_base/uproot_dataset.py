from root_gnn_base import dataset
import torch
import uproot
import glob
import os
import awkward as ak
import numpy as np
import time

def node_features_from_ak(ch, node_branch_names, node_branch_types, node_feature_scales):
    node_types = []
    n_types = len(node_branch_names[0])
    for i in range(n_types):
        features = []
        branch_type = node_branch_types[i]
        for j in range(len(node_branch_names)):
            if node_branch_names[j] == 'CALC_E':
                features.append(features[0] * np.cosh(features[1]))
            elif node_branch_names[j] == 'NODE_TYPE':
                features.append(ak.full_like(features[0], i))
            elif isinstance(node_branch_names[j][i], str):
                features.append(ch[node_branch_names[j][i]])
            elif isinstance(node_branch_names[j][i], (int, float)):
                features.append(ak.full_like(features[0], node_branch_names[j][i]))
        if branch_type == 'single':
            features = [f[:,np.newaxis] for f in features]
        node_types.append(ak.Array(features))
    node_features = ak.concatenate(node_types, axis=2) * node_feature_scales #axis order at this point is (feature, event, node)
    return node_features

class UprootDataset(dataset.RootDataset):
    def process(self):
        starttime = time.time()
        self.files = glob.glob(os.path.join(self.raw_dir, self.file_names))
        branches = self.get_list_of_branches()
        self.chain = uproot.concatenate([f + ':' + self.tree_name for f in self.files], branches, num_workers=4)
        node_features = node_features_from_ak(self.chain, self.node_branch_names, self.node_branch_types, self.node_feature_scales)
        loadtime = time.time()
        n_nodes = ak.num(node_features[0], axis=1) #number of nodes for each event
        ftime = time.time()
        self.graphs = [dataset.full_connected_graph(n, False) for n in n_nodes]
        itime = time.time()
        for i in range(len(self.graphs)):
            if i % 10000 == 0:
                print(f'Processing event {i}/{len(self.graphs)}')
            self.graphs[i].ndata['features'] = torch.transpose(torch.tensor(node_features[:,i,:]),0,1).to(torch.float)
        self.label = torch.stack([torch.full((len(self.graphs),),torch.tensor(self.label)), torch.tensor(ak.values_astype(self.chain[self.fold_var], np.int64))], dim=1)
        gtime = time.time()
        print()
        print(f'load time: {loadtime - starttime} s')
        print(f'feature time: {ftime - loadtime} s')
        print(f'graph time: {itime - ftime} s')
        print(f'graph data time: {gtime - itime} s')

