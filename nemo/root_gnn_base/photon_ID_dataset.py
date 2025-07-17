from root_gnn_base import dataset
import dgl
import torch
import numpy as np

def radius_graph(features, radii, self_loops=False):
    senders = []
    receivers = []
    n_nodes = features.shape[0]
    senders = np.arange(n_nodes*n_nodes) // n_nodes
    receivers = np.arange(n_nodes*n_nodes) % n_nodes
    if not self_loops and n_nodes > 1:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]
    for k, r in radii.items():
        d = features[senders, k] - features[receivers, k]
        mask = np.abs(d) < r
        senders = senders[mask]
        receivers = receivers[mask]
    return dgl.graph((senders, receivers))

class PhotonIDDataset(dataset.LazyMultiLabelDataset):
    def __init__(self, eta_radius, phi_radius, **kwargs):
        self.eta_radius = eta_radius
        self.phi_radius = phi_radius
        super().__init__(**kwargs)
    def make_graph(self, ch):
        features, _ = dataset.node_features_from_tree(ch, self.node_branch_names, self.node_branch_types, self.node_feature_scales)
        features = features[features[:,0] != 0]
        #Delta Eta, Delta Phi, Adjacent Layer
        g = radius_graph(features, {1: self.eta_radius, 2: self.phi_radius, 6: 1.1}, self_loops=True) #Self loops ensure last cell is included even if disconnected
        g.ndata['features'] = features
        u, v = g.edges()
        deta = features[u, 1] - features[v, 1]
        dphi = g.ndata['features'][u, 2] - g.ndata['features'][v, 2]
        dphi = torch.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dR   = torch.sqrt(deta**2 + dphi**2)
        dx   = features[u, 3] - features[v, 3]
        dy   = features[u, 4] - features[v, 4]
        dz   = features[u, 5] - features[v, 5]
        g.edata['features'] = torch.stack([deta, dphi, dR, dx, dy, dz], dim=1)
        return g