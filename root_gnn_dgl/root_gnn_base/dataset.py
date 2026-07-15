from dgl.data import DGLDataset
import dgl
import uproot
import awkward as ak
import torch
import os
import glob
import time
import numpy as np
import re
from root_gnn_base import utils

FEATURE_DTYPE = torch.float32

def node_features_from_tree(ch, node_branch_names, node_branch_types, node_feature_scales):
    lengths = []
    for branch, node_type in zip(node_branch_names[0], node_branch_types):
        if node_type == 'single':
            lengths.append(1)
        elif node_type == 'vector':
            lengths.append(len(ch[branch]))
        else:
            print('Unknown node branch type: {}'.format(node_type))
    features = []
    for node_feat in node_branch_names:
        if node_feat == 'CALC_E':
            features.append(features[0]*torch.cosh(features[1]))
            continue
        elif node_feat == 'NODE_TYPE':
            feat = []
            for i, length in enumerate(lengths):
                feat.extend([i,]*length)
            features.append(torch.tensor(feat, dtype=FEATURE_DTYPE))
            continue
        feat = []
        itype = 0
        for length, branch, node_type in zip(lengths, node_feat, node_branch_types):
            if isinstance(branch, (int, float, complex)):
                feat.extend([branch,]*length)
            elif branch == 'CALC_E':
                this_type_starts_at = sum(lengths[:itype])
                this_type_ends_at = sum(lengths[:itype+1])
                feat.extend(features[0][this_type_starts_at:this_type_ends_at]*torch.cosh(features[1][this_type_starts_at:this_type_ends_at]))
            elif node_type == 'single':
                feat.append(ch[branch])
            elif node_type == 'vector':
                feat.extend(ch[branch])
            itype += 1
        features.append(torch.as_tensor(np.asarray(feat, dtype=np.float32), dtype=FEATURE_DTYPE))
    return torch.stack(features, dim=1) * node_feature_scales, lengths

def full_connected_graph(n_nodes, self_loops=True):
    senders = np.arange(n_nodes*n_nodes) // n_nodes
    receivers = np.arange(n_nodes*n_nodes) % n_nodes
    if not self_loops and n_nodes > 1:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]
    return dgl.graph((senders, receivers))

def _selection_context(ch):
    if isinstance(ch, dict):
        return ch
    if hasattr(ch, 'GetListOfBranches'):
        context = {}
        for branch in ch.GetListOfBranches():
            branch_name = branch.GetName()
            try:
                context[branch_name] = getattr(ch, branch_name)
            except Exception:
                continue
        return context
    return vars(ch)

def check_selection(ch, selection):
    if isinstance(selection, str):
        return bool(eval(selection, {"__builtins__": {}}, _selection_context(ch)))
    var, cut, op = selection
    if op == '>':
        return ch[var] > cut
    elif op == '>=':
        return ch[var] >= cut
    elif op == '<':
        return ch[var] < cut
    elif op == '<=':
        return ch[var] <= cut
    elif op == '==':
        return ch[var] == cut
    elif op == '!=':
        return ch[var] != cut
    raise ValueError(f'Unknown selection operator: {op}')

def check_selections(ch, selections):
    for selection in selections:
        if not check_selection(ch, selection):
            return False
    return True

def selection_branches(selection):
    if isinstance(selection, str):
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", selection)
        keywords = {"and", "or", "not", "True", "False"}
        return [tok for tok in tokens if tok not in keywords]
    if isinstance(selection, (list, tuple)) and len(selection) > 0:
        return [selection[0]]
    return []

def init_cutflow(selections):
    return {"selections": list(selections), "counts": [0] * len(selections), "total": 0}

def update_cutflow(cutflow, ch):
    cutflow["total"] += 1
    passed = True
    for i, selection in enumerate(cutflow["selections"]):
        if passed and check_selection(ch, selection):
            cutflow["counts"][i] += 1
        else:
            passed = False

def print_cutflow(cutflow, title="Cutflow"):
    total = cutflow["total"]
    print(f"\n{title}")
    print(f"{'Step':<6} {'Label':<30} {'Yield':>12} {'Ind. Eff.':>12} {'Tot. Eff.':>12}")
    prev = total
    current = total
    print(f"{0:<6} {'all events':<30} {total:>12} {1.0:>12.4f} {1.0:>12.4f}")
    for i, (selection, count) in enumerate(zip(cutflow["selections"], cutflow["counts"]), start=1):
        current = count
        ind_eff = current / prev if prev else 0.0
        tot_eff = current / total if total else 0.0
        print(f"{i:<6} {str(selection):<30} {current:>12} {ind_eff:>12.4f} {tot_eff:>12.4f}")
        prev = current

def selection_mask(data, selections):
    if len(selections) == 0:
        first_field = data.fields[0] if len(data.fields) > 0 else None
        return np.ones(len(data[first_field]), dtype=bool) if first_field is not None else None
    mask = None
    for selection in selections:
        current_mask = eval(selection, {"__builtins__": {}}, data) if isinstance(selection, str) else check_selection(data, selection)
        if mask is None:
            mask = current_mask
        else:
            mask = mask & current_mask
    return mask

def compute_cutflow(data, selections):
    cutflow = init_cutflow(selections)
    first_field = data.fields[0] if len(data.fields) > 0 else None
    cutflow["total"] = len(data[first_field]) if first_field is not None else 0
    if first_field is None:
        return cutflow
    running_mask = np.ones(cutflow["total"], dtype=bool)
    for i, selection in enumerate(selections):
        current_mask = eval(selection, {"__builtins__": {}}, data) if isinstance(selection, str) else check_selection(data, selection)
        running_mask = running_mask & current_mask
        cutflow["counts"][i] = int(ak.sum(running_mask))
    return cutflow

class RootDataset(DGLDataset):
    def __init__(self, name=None, raw_dir=None, save_dir=None, label=1, file_names = '*.root', node_branch_names=None, node_branch_types=None, node_feature_scales=None, 
                 selections=[], save=True, tree_name = 'nominal_Loose', fold_var = 'eventNumber', weight_var = None, chunks = 1, process_chunks = None, global_features = [], tracking_info = [], **kwargs):
        ignored_keys = {'shuffle_chunks', 'shuffle_seed', 'batch_size', 'padding_mode', 'folding'}
        unused_kwargs = {k: v for k, v in kwargs.items() if k not in ignored_keys}
        if len(unused_kwargs) > 0:
            print(f'Unused args while creating RootDataset: {unused_kwargs}')
        self.label = label
        self.counts = []
        self.selections = selections
        self.save_to_disk = save
        self.file_names = file_names
        self.node_branch_names = node_branch_names
        self.node_branch_types = node_branch_types
        self.node_feature_scales = torch.tensor([float(sf) for sf in node_feature_scales], dtype=FEATURE_DTYPE)
        self.tree_name = tree_name
        self.fold_var = fold_var
        self.tracking_info = tracking_info
        self.tracking_info.insert(0, fold_var)
        if weight_var is None:
            weight_var = 1
        self.tracking_info.insert(1, weight_var)
        self.global_features = global_features
        self.chunks = chunks
        self.process_chunks = process_chunks
        if self.process_chunks is None:
            self.process_chunks = [i for i in range(self.chunks)]
        self.times = [0, 0]
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)

    def get_list_of_branches(self):
        branches = []
        for feat in self.node_branch_names:
            if isinstance(feat, list):
                for branch in feat:
                    if branch == 'CALC_E':
                        continue
                    if isinstance(branch, str):
                        branches.append(branch)
        for feat in self.global_features:
            if isinstance(feat, str):
                branches.append(feat)
        for feat in self.tracking_info:
            if isinstance(feat, str):
                branches.append(feat)
        for selection in self.selections:
            branches.extend(selection_branches(selection))
        return list(set(branches))  # Remove duplicates

    def make_graph(self, ch):
        t1 = time.time()
        features, _ = node_features_from_tree(ch, self.node_branch_names, self.node_branch_types, self.node_feature_scales)
        features = features[features[:,0] != 0]
        t2 = time.time()
        g = full_connected_graph(features.shape[0], self_loops=False)
        g.ndata['features'] = features
        t3 = time.time()
        self.times[0] += t2 - t1
        self.times[1] += t3 - t2
        return g

    def process(self):
        times = [0, 0, 0]
        oldtime = time.time()
        if isinstance(self.file_names, str):
            self.files = glob.glob(os.path.join(self.raw_dir, self.file_names))
        else:
            self.files = []
            for file_name in self.file_names:
                self.files.extend(glob.glob(os.path.join(self.raw_dir, file_name)))
        branches = self.get_list_of_branches()

        # Read all files and concatenate arrays
        arrays = []
        for file in self.files:
            with uproot.open(file) as f:
                arrays.append(f[self.tree_name].arrays(branches, library="ak"))
        if len(arrays) == 0:
            print('No files found in {}'.format(os.path.join(self.raw_dir, self.file_names)))
            return
        data = ak.concatenate(arrays, axis=0)
        sel_mask = selection_mask(data, self.selections)
        selected_indices = np.arange(len(data[branches[0]]))
        if sel_mask is not None:
            selected_indices = selected_indices[ak.to_numpy(sel_mask)]
        n_entries = len(selected_indices)
        newtime = time.time()
        times[0] += newtime - oldtime
        chunks = np.array_split(selected_indices, self.chunks)
        chunks = [chunk for i, chunk in enumerate(chunks) if i in self.process_chunks]

        self.graph_chunks = []
        self.label_chunks = []
        self.tracking_chunks = []
        self.global_chunks = []
        cutflow = init_cutflow(self.selections)
        chunk_id = -1
        for chunk in chunks:
            print('Processing chunk {}/{}'.format(chunk_id + 1, len(chunks)), flush=True)
            chunk_id += 1
            graphs = []
            labels = []
            tracking = []
            globals = []
            for ientry in chunk:
                if (ientry % 10000 == 0):
                    print('Processing event {}/{}'.format(ientry, n_entries), flush=True)
                ch = {b: data[b][ientry] for b in branches}
                passed = True
                for selection in self.selections:
                    if not check_selection(ch, selection):
                        passed = False
                        continue
                oldtime = newtime
                newtime = time.time()
                times[1] += newtime - oldtime
                if passed:
                    graphs.append(self.make_graph(ch))
                    labels.append(self.label)
                    tracking.append(torch.zeros(len(self.tracking_info), dtype=FEATURE_DTYPE))
                    globals.append(torch.zeros(len(self.global_features), dtype=FEATURE_DTYPE))
                    for i_ti, tr_branch in enumerate(self.tracking_info):
                        if isinstance(tr_branch, str):
                            dtype = tracking[-1].dtype
                            tracking[-1][i_ti] = torch.as_tensor(ch[tr_branch], dtype=dtype)
                            # tracking[-1][i_ti] = ch[tr_branch]
                        else:
                            tracking[-1][i_ti] = tr_branch
                    for i_gl, gl_branch in enumerate(self.global_features):
                        globals[-1][i_gl] = ch[gl_branch]
                oldtime = newtime
                newtime = time.time()
                times[2] += newtime - oldtime

            labels = torch.tensor(labels)
            tracking = torch.stack(tracking)
            globals = torch.stack(globals)

            # self.graph_chunks.append(graphs)
            # self.label_chunks.append(labels)
            # self.tracking_chunks.append(tracking)
            # self.global_chunks.append(globals)
            # self.counts.append(len(graphs))
            
            if (self.chunks > 1):
                self.save_chunk(chunk_id, graphs, labels, tracking, globals)
            else:
                self.labels = labels
                self.tracking = tracking
                self.global_features = globals
                self.graphs = graphs
                self.save()
        return

    def save(self):
        if not self.save_to_disk:
            return
        if self.chunks > 1 and (not hasattr(self, "graph_chunks") or len(self.graph_chunks) == 0):
            print(f"Skipping save for {self.name}: no graphs were created.")
            return
        graph_path = os.path.join(self.save_dir, self.name + '.bin')
        if self.chunks == 1:
            print(f'Saving dataset to {os.path.join(self.save_dir, self.name + ".bin")}')
            dgl.save_graphs(str(graph_path), self.graphs, {'labels': torch.tensor(self.labels), 'tracking': torch.tensor(self.tracking), 'global': torch.tensor(self.global_features)})
        else:
            for i in range(len(self.process_chunks)):
                print(f'Saving dataset to {os.path.join(self.save_dir, self.name + f"_{self.process_chunks[i]}.bin")}')

                dgl.save_graphs(str(graph_path).replace('.bin', f'_{self.process_chunks[i]}.bin'), self.graph_chunks[i], {'labels': self.label_chunks[i], 'tracking': self.tracking_chunks[i], 'global': self.global_chunks[i]})

    def save_chunk(self, chunk_id, graphs, labels, tracking, globals):
        if not self.save_to_disk:
            return
        if len(graphs) == 0:
            print(f"Skipping save for chunk {chunk_id}: no graphs passed selections.")
            return
        graph_path = os.path.join(self.save_dir, self.name + '.bin')
        print(f'Saving dataset to {os.path.join(self.save_dir, self.name + f"_{self.process_chunks[chunk_id]}.bin")}')
        dgl.save_graphs(str(graph_path).replace('.bin', f'_{self.process_chunks[chunk_id]}.bin'), graphs, {'labels': labels, 'tracking': tracking, 'global': globals})

    def has_cache(self):
        print(f'Checking for cache of {self.name}')
        if not self.save_to_disk:
            print('Skipping load.')
            return False
        if self.chunks == 1:
            graph_path = os.path.join(self.save_dir, self.name + '.bin')
            return os.path.exists(graph_path)
        else:
            for i in range(len(self.process_chunks)):
                graph_path = os.path.join(self.save_dir, self.name + f'_{self.process_chunks[i]}.bin')
                if not os.path.exists(graph_path):
                    print(f'File {graph_path} does not exist, processing.')
                    return False
            return True

    def load(self):
        if self.chunks == 1:
            print(f'Loading dataset from {os.path.join(self.save_dir, self.name + ".bin")}')
            graphs, label_dict = dgl.load_graphs(os.path.join(self.save_dir, self.name + '.bin'))
            self.graphs = graphs
            self.labels = label_dict['labels']
            self.tracking = label_dict['tracking']
            self.global_features = label_dict['global']
        else:
            self.graphs = []
            self.labels = []
            self.tracking = []
            self.global_features = []
            for i in range(self.chunks):
                try:
                    print(f'Loading dataset from {os.path.join(self.save_dir, self.name + f"_{self.process_chunks[i]}.bin")}')
                    graphs, label = dgl.load_graphs(os.path.join(self.save_dir, self.name + f'_{self.process_chunks[i]}.bin'))
                    self.graphs.extend(graphs)
                    self.labels.append(label['labels'])
                    self.tracking.append(label['tracking'])
                    self.global_features.append(label['global'])
                except Exception as e:
                    print(e)
            self.labels = torch.cat(self.labels)
            self.tracking = torch.cat(self.tracking)
            self.global_features = torch.cat(self.global_features)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.tracking[idx], self.global_features[idx]
    
    def __len__(self):
        return len(self.graphs)

#Dataset with edge features added (deta, dphi, dR)
class EdgeDataset(RootDataset):
    def make_graph(self, ch):
        g = super().make_graph(ch)
        u, v = g.edges()
        deta = g.ndata['features'][u, 1] - g.ndata['features'][v, 1]
        dphi = g.ndata['features'][u, 2] - g.ndata['features'][v, 2]
        dphi = torch.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dR   = torch.sqrt(deta**2 + dphi**2)
        g.edata['features'] = torch.stack([deta, dphi, dR], dim=1)
        return g

class tHbbEdgeDataset(RootDataset):
    def __init__(self, exclude_branches=None, **kwargs):
        self.exclude_branches = exclude_branches
        super().__init__(**kwargs)

    def get_list_of_branches(self):
        br = super().get_list_of_branches()
        for sector in self.exclude_branches:
            if sector == None:
                continue
            for excl in sector:
                if type(excl) == str:
                    br.append(excl)
        return br

    def make_graph(self, ch):
        features, lengths = node_features_from_tree(ch, self.node_branch_names, self.node_branch_types, self.node_feature_scales)

        include_mask = torch.ones(features.shape[0], dtype=torch.bool)
        node_idx = 0
        for sector, length in zip(self.exclude_branches, lengths):
            if sector == None:
                node_idx += length
                continue
            for excl in sector:
                if type(excl) == int:
                    include_mask[excl + node_idx] = False
                elif type(excl) == str:
                    include_mask[getattr(self.chain, excl) + node_idx] = False
        g = full_connected_graph(features[include_mask].shape[0], self_loops=False)
        g.ndata['features'] = features[include_mask]

        u, v = g.edges()
        deta = g.ndata['features'][u, 1] - g.ndata['features'][v, 1]
        dphi = g.ndata['features'][u, 2] - g.ndata['features'][v, 2]
        dphi = torch.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dR   = torch.sqrt(deta**2 + dphi**2)
        g.edata['features'] = torch.stack([deta, dphi, dR], dim=1)
        return g
    
class LazyDataset(EdgeDataset):
    def __init__(self, buffer_size = 2, **kwargs):
        self.buffer = [None,] * buffer_size
        self.buffer_ptr = 0
        self.get_item_calls = 0
        self.buffer_indices = [-1,] * buffer_size
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        self.get_item_calls += 1
        chunk_idx = -1
        sum = 0
        ev_idx = -999
        for i, count in enumerate(self.counts):
            sum += count
            if idx < sum:
                chunk_idx = i
                ev_idx = idx - sum + count
                break
        buf_idx = self.buffer_get(chunk_idx)
        if ev_idx >= len(self.buffer[buf_idx][0]):
            print(f'Getting event {ev_idx} from chunk {chunk_idx} from buffer {buf_idx}. Calls: {self.get_item_calls}')
            print(len(self.buffer))
            print(self.counts)
            print(len(self.buffer[buf_idx][0]))
        return self.buffer[buf_idx][0][ev_idx], self.buffer[buf_idx][1]['labels'][ev_idx], self.buffer[buf_idx][1]['tracking'][ev_idx], self.buffer[buf_idx][1]['global'][ev_idx]

    def buffer_get(self, buffer_idx):
        if buffer_idx in self.buffer_indices:
            for i in range(len(self.buffer)):
                if self.buffer_indices[i] == buffer_idx:
                    return i
        else:
            print(f'Loading dataset from {os.path.join(self.save_dir, self.name + f"_{buffer_idx}.bin")}', flush=True)
            self.buffer_ptr = (self.buffer_ptr + 1) % len(self.buffer)
            self.buffer[self.buffer_ptr] = dgl.load_graphs(os.path.join(self.save_dir, self.name + f'_{buffer_idx}.bin'))
            self.buffer_indices[self.buffer_ptr] = buffer_idx
            return self.buffer_ptr

    def load(self):
        self.counts = []
        self.tracking = []
        try:
            for i in range(self.chunks):
                print(f'Loading dataset from {os.path.join(self.save_dir, self.name + f"_{self.process_chunks[i]}.bin")}')
                l = dgl.data.graph_serialize.load_labels_v2(os.path.join(self.save_dir, self.name + f'_{self.process_chunks[i]}.bin'))
                self.counts.append(len(l['tracking']))
                self.tracking.append(l['tracking'])
            self.tracking = torch.cat(self.tracking)
        except Exception as e:
            print(e)

    def __len__(self):
        return sum(self.counts)

class MultiLabelDataset(EdgeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_list_of_branches(self):
        br =  super().get_list_of_branches()
        for l in self.label:
            if isinstance(l, str):
                br.append(l)
            if isinstance(l, dict):
                br.append(l['branch'])
        return br

    def get_label(self, ch):
        label = []
        for l in self.label:
            if isinstance(l, str):
                label.append((getattr(ch, l)))
            if isinstance(l, dict):
                label.append(getattr(ch, l['branch'])*float(l['scale']))
            if isinstance(l, float) or isinstance(l, int):
                label.append(l)

        return torch.tensor(label)

    def process(self):
        times = [0, 0, 0]
        oldtime = time.time()
        if isinstance(self.file_names, str):
            self.files = glob.glob(os.path.join(self.raw_dir, self.file_names))
        else:
            self.files = []
            for file_name in self.file_names:
                self.files.extend(glob.glob(os.path.join(self.raw_dir, file_name)))
        self.chain = ROOT.TChain(self.tree_name)
        if len(self.files) == 0:
            print('No files found in {}'.format(os.path.join(self.raw_dir, self.file_names)))
        for file in self.files:
            utils.set_timeout(60*2)
            self.chain.Add(file)
            utils.unset_timeout()
        branches = self.get_list_of_branches()
        self.chain.SetBranchStatus('*', 0)
        for branch in branches:
            self.chain.SetBranchStatus(branch, 1)
        newtime = time.time()
        times[0] += newtime - oldtime
        all_indices = np.arange(self.chain.GetEntries())
        branches = self.get_list_of_branches()
        data = {branch: [] for branch in branches}
        for ientry in all_indices:
            self.chain.GetEntry(ientry)
            for branch in branches:
                data[branch].append(getattr(self.chain, branch))
        data = {branch: ak.Array(values) for branch, values in data.items()}
        sel_mask = selection_mask(data, self.selections)
        selected_indices = all_indices[ak.to_numpy(sel_mask)] if sel_mask is not None else all_indices
        chunks = np.array_split(selected_indices, self.chunks)
        chunks = [chunk for i, chunk in enumerate(chunks) if i in self.process_chunks]
        self.graph_chunks = []
        self.label_chunks = []
        self.tracking_chunks = []
        self.global_chunks = []
        chunk_id = -1
        for chunk in chunks:
            chunk_id += 1
            graphs = []
            labels = []
            tracking = []
            globals = []
            for ientry in chunk:
                if (ientry % 10000 == 0):
                    print('Processing event {}/{}'.format(ientry, self.chain.GetEntries()), flush=True)
                self.chain.GetEntry(ientry)
                ch = {b: getattr(self.chain, b) for b in branches}
                passed = True
                for selection in self.selections:
                    if not check_selection(self.chain, selection):
                        passed = False
                        continue
                oldtime = newtime
                newtime = time.time()
                times[1] += newtime - oldtime
                if passed:
                    graphs.append(self.make_graph(self.chain))
                    labels.append(self.get_label(self.chain))
                    tracking.append(torch.zeros(len(self.tracking_info), dtype=FEATURE_DTYPE))
                    globals.append(torch.zeros(len(self.global_features), dtype=FEATURE_DTYPE))
                    for i_ti, tr_branch in enumerate(self.tracking_info):
                        if isinstance(tr_branch, str):
                            tracking[-1][i_ti] = getattr(self.chain, tr_branch)
                        else:
                            tracking[-1][i_ti] = tr_branch
                    for i_gl, gl_branch in enumerate(self.global_features):
                        globals[-1][i_gl] = getattr(self.chain, gl_branch)
                oldtime = newtime
                newtime = time.time()
                times[2] += newtime - oldtime

            labels = torch.stack(labels)
            self.save_chunk(chunk_id, graphs, labels, torch.stack(tracking), torch.stack(globals))
            # self.graph_chunks.append(graphs)
            # self.label_chunks.append(labels)
            # self.tracking_chunks.append(torch.stack(tracking))
            # self.global_chunks.append(torch.stack(globals))
            # self.counts.append(len(graphs))
        return
        self.graphs = self.graph_chunks[0]
        for chunk in self.graph_chunks[1:]:
            self.graphs += chunk

        self.labels = torch.cat(self.label_chunks)
        self.tracking = torch.cat(self.tracking_chunks)
        self.global_features = torch.cat(self.global_chunks)
        print('Time spent: Creating TChain: {}s, Getting Entries and Selection: {}s, Graph Creation: {}s'.format(*times))
        print('Time spent in node_features_from_tree: {}s, full_connected_graph: {}s'.format(*self.times))

class LazyMultiLabelDataset(MultiLabelDataset, LazyDataset):
    def __init__(self, buffer_size = 2, **kwargs):
        LazyDataset.__init__(self, buffer_size=buffer_size, **kwargs)

class MultiLabeltHbbDataset(MultiLabelDataset, tHbbEdgeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_list_of_branches(self):
        br =  super().get_list_of_branches()
        for sector in self.exclude_branches:
            if sector == None:
                continue
            for excl in sector:
                if type(excl) == str:
                    br.append(excl)
        return br


class AugmentedDataset(RootDataset):

    def __init__(self, seed = 2, feature_index = None, node_mapping = None, **kwargs):
        self.seed = seed
        np.random.seed(seed)
        if(feature_index == None):
            self.feature_index = {"pt": 0, "eta": 1, "phi": 2, "energy": 3, "btag": 4, "charge": 5, "node_type": 6}
        if (node_mapping == None):
            self.node_mapping = {"jet": 0, "ele": 1, "mu": 2, "ph": 3, "MET": 4}
        super().__init__(**kwargs)

    def detector_noise(self, node_features):
        noise = np.zeros_like(node_features)
        
        node_types = node_features[:, self.feature_index["node_type"]]
        pts = node_features[:, self.feature_index["pt"]]
        etas = node_features[:, self.feature_index["eta"]]
        energies = node_features[:, self.feature_index["energy"]]

        # Noise calculation for jets
        jet_mask = (node_types == self.node_mapping["jet"])
        jet_pts = pts[jet_mask]
        jet_etas = etas[jet_mask]

        if (jet_mask.sum() > 0):
            jet_resolutions = np.where(
                jet_pts <= 0.1, 0.0,
                np.where(
                    np.abs(jet_etas) <= 0.5, np.sqrt(0.06**2 + jet_pts**2 * 1.3e-3**2),
                    np.where(
                        np.abs(jet_etas) <= 1.5, np.sqrt(0.10**2 + jet_pts**2 * 1.7e-3**2),
                        np.where(
                            np.abs(jet_etas) <= 2.5, np.sqrt(0.25**2 + jet_pts**2 * 3.1e-3**2),
                            0.0
                        )
                    )
                )
            )
            noise[jet_mask, self.feature_index["pt"]] = np.random.normal(loc=0.0, scale=jet_resolutions)

        # Noise calculation for electrons
        ele_mask = (node_types == self.node_mapping["ele"])
        ele_pts = pts[ele_mask]
        ele_etas = etas[ele_mask]

        if (ele_mask.sum() > 0):
            ele_resolutions = np.where(
                np.abs(ele_etas) <= 0.5, np.sqrt(0.03**2 + ele_pts**2 * 1.3e-3**2),
                np.where(
                    np.abs(ele_etas) <= 1.5, np.sqrt(0.05**2 + ele_pts**2 * 1.7e-3**2),
                    np.where(
                        np.abs(ele_etas) <= 2.5, np.sqrt(0.15**2 + ele_pts**2 * 3.1e-3**2),
                        0.0
                    )
                )
            )
            noise[ele_mask, self.feature_index["pt"]] = np.random.normal(loc=0.0, scale=ele_resolutions)

        # Noise calculation for muons
        mu_mask = (node_types == self.node_mapping["mu"])
        mu_pts = pts[mu_mask]
        mu_etas = etas[mu_mask]

        if (mu_mask.sum() > 0):
            mu_resolutions = np.where(
                np.abs(mu_etas) <= 0.5, np.sqrt(0.01**2 + mu_pts**2 * 1.0e-4**2),
                np.where(
                    np.abs(mu_etas) <= 1.5, np.sqrt(0.015**2 + mu_pts**2 * 1.5e-4**2),
                    np.where(
                        np.abs(mu_etas) <= 2.5, np.sqrt(0.025**2 + mu_pts**2 * 3.5e-4**2),
                        0.0
                    )
                )
            )
            noise[mu_mask, self.feature_index["pt"]] = np.random.normal(loc=0.0, scale=mu_resolutions)

        # Noise calculation for photons
        ph_mask = (node_types == self.node_mapping["ph"])
        ph_etas = etas[ph_mask]
        ph_energies = energies[ph_mask]

        if (ph_mask.sum() > 0):
            ph_resolutions = np.where(
                np.abs(ph_etas) <= 3.2, np.sqrt(ph_energies**2 * 0.0017**2 + ph_energies * 0.101**2),
                np.where(
                    np.abs(ph_etas) <= 4.9, np.sqrt(ph_energies**2 * 0.0350**2 + ph_energies * 0.285**2),
                    0.0
                )
            )
            noise[ph_mask, self.feature_index["energy"]] = np.random.normal(loc=0.0, scale=ph_resolutions)
        return noise
    
    def make_graph(self, ch):
        g = super().make_graph(ch)

        g.ndata['augmented_features'] = g.ndata['features']

        num_nodes = len(g.ndata['features'][:, 0])

        # Rotations: phi -> phi + delta_phi
        phi_index = self.feature_index["phi"]
        # Generate a single delta_phi for all nodes
        delta_phi = np.random.uniform(low=-np.pi, high=np.pi)
        
        # Apply the same delta_phi to all nodes
        g.ndata['augmented_features'][:, phi_index] = (g.ndata['augmented_features'][:, phi_index] + delta_phi + np.pi) % (2 * np.pi) - np.pi

        # Reflections: eta -> -1 * eta, phi -> -1 * phi
        eta_index = self.feature_index["eta"]

        eta_reflection = np.random.choice([-1, 1])
        phi_reflection = np.random.choice([-1, 1])

        g.ndata['augmented_features'][:, eta_index] = g.ndata['augmented_features'][:, eta_index] * eta_reflection
        g.ndata['augmented_features'][:, phi_index] = g.ndata['augmented_features'][:, phi_index] * phi_reflection


        # Detector Noise: pt -> pt + normal(pt, noise(pt))
        noise = self.detector_noise(g.ndata['augmented_features'])
        g.ndata['augmented_features'] = g.ndata['augmented_features'] + noise

        pt_index = self.feature_index["pt"]
        if (g.ndata['augmented_features'][-1][self.feature_index["node_type"]] == self.node_mapping["MET"]):
            # Initialize sums of px and py
            sum_px = 0
            sum_py = 0

            # Loop over all nodes except the last one (MET node)
            for i in range(len(g.ndata['augmented_features']) - 1):
                pt = g.ndata['augmented_features'][i][pt_index]
                phi = g.ndata['augmented_features'][i][phi_index]

                # Compute px and py
                px = pt * np.cos(phi)
                py = pt * np.sin(phi)

                # Sum px and py
                sum_px += px
                sum_py += py

            # Calculate MET
            g.ndata['augmented_features'][-1][pt_index] = np.sqrt(sum_px**2 + sum_py**2)

        u, v = g.edges()
        deta = g.ndata['features'][u, 1] - g.ndata['features'][v, 1]
        dphi = g.ndata['features'][u, 2] - g.ndata['features'][v, 2]
        dphi = torch.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dR   = torch.sqrt(deta**2 + dphi**2)
        g.edata['features'] = torch.stack([deta, dphi, dR], dim=1)

        deta = g.ndata['augmented_features'][u, 1] - g.ndata['augmented_features'][v, 1]
        dphi = g.ndata['augmented_features'][u, 2] - g.ndata['augmented_features'][v, 2]
        dphi = torch.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = torch.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        dR   = torch.sqrt(deta**2 + dphi**2)
        g.edata['augmented_features'] = torch.stack([deta, dphi, dR], dim=1)

        return g
