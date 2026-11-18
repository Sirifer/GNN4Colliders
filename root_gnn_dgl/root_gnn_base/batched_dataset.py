from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from dgl.data import DGLDataset
import torch
import time
import os
import dgl
from root_gnn_base import utils

def GetBatchedLoader(dataset, batch_size, mask_fn = None, drop_last=True, **kwargs):
    if mask_fn == None:
        mask_fn = lambda x: torch.ones(len(x), dtype=torch.bool)
    dloader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(torch.arange(len(dataset))[mask_fn(dataset)]), batch_size=batch_size, drop_last=drop_last, num_workers = 0)
    return dloader

class FixedOrderSampler:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)

def hash_partition(indices, chunks, seed):
    if chunks <= 1:
        return torch.zeros(len(indices), dtype=torch.long)
    hashed = (indices * 1103515245 + int(seed)) % 2147483647
    return hashed % chunks

#Dataset which contains prebatched shuffled graphs. Cannot be saved to disk, else batching info is lost.
class PreBatchedDataset(DGLDataset):
    def __init__(self, start_dataset, batch_size, mask_fn = None, drop_last=True, save_to_disk = True, suffix = '', chunks = 1, chunkno = -1, shuffle = True, padding_mode = 'NONE', hidden_size=64, shuffle_seed=12345, **kwargs):
        print(f'Unused kwargs: {kwargs}')
        self.start_dataset = start_dataset
        self.start_dataset.load()
        
        self.batch_size = batch_size
        self.chunks = chunks
        self.chunkno = chunkno
        self.mask_fn = mask_fn
        if self.mask_fn == None:
            self.mask_fn = lambda x: torch.ones(len(x), dtype=torch.bool)
        self.drop_last = drop_last
        self.graphs = []
        self.label = []
        self.padding_mode = padding_mode
        self.save_to_disk = save_to_disk
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.suffix = suffix
        self.current_chunk = None
        self.current_chunk_idx = -1
        self.hid_size = hidden_size
        super().__init__(name = start_dataset.name + '_prebatched_padded', save_dir=start_dataset.save_dir)

    def process(self):
        dataset_indices = torch.arange(len(self.start_dataset))
        fold_indices = dataset_indices[self.mask_fn(self.start_dataset)]
        if self.shuffle:
            if self.chunks > 1 and self.chunkno >= 0:
                partitions = hash_partition(fold_indices, self.chunks, self.shuffle_seed)
                fold_indices = fold_indices[partitions == self.chunkno]
            print(f'Processing shuffled chunk {self.chunkno} of {self.chunks} with {len(fold_indices)} events from {len(self.start_dataset)} total')
            dloader = GraphDataLoader(self.start_dataset, sampler=FixedOrderSampler(fold_indices), batch_size=self.batch_size, drop_last=self.drop_last)
        else: #Only don't shuffle if we're doing inference. Then we want all of the events anyways?
            print(f'Processing sequential dataset with {len(self.start_dataset)} events')
            dloader = GraphDataLoader(self.start_dataset, sampler=SequentialSampler(self.start_dataset), batch_size=self.batch_size, drop_last=self.drop_last)
        self.graphs = []
        self.labels = []
        self.tracking = []
        self.globals = []
        self.batch_num_nodes = []
        self.batch_num_edges = []
        max_edges = 0
        max_nodes = 0
        load_batch_start = time.time()
        for batch, label, tracking, global_feat in dloader:
            if batch.num_edges() > max_edges:
                max_edges = batch.num_edges()
            if batch.num_nodes() > max_nodes:
                max_nodes = batch.num_nodes()
            self.graphs.append(batch)
            self.labels.append(label)
            self.tracking.append(tracking)
            self.globals.append(global_feat)
        load_batch_end = time.time()
        print(f'Loaded {len(self.graphs)} batches in {load_batch_end - load_batch_start} seconds')
        if self.shuffle and len(self.graphs) > 1:
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + 7919 * max(self.chunkno, 0))
            batch_order = torch.randperm(len(self.graphs), generator=generator).tolist()
            self.graphs = [self.graphs[i] for i in batch_order]
            self.labels = [self.labels[i] for i in batch_order]
            self.tracking = [self.tracking[i] for i in batch_order]
            self.globals = [self.globals[i] for i in batch_order]
            print(f'Shuffled {len(self.graphs)} prebatched batches before padding')
        if self.padding_mode == 'STEPS':
            pad_node, pad_edge = utils.pad_size(self.batch_size, max_edges, max_nodes)
        elif self.padding_mode == 'FIXED':
            print('Padding to fixed size. This is currently hardcoded.')
            pad_node = 16000
            pad_edge = 104000
        elif self.padding_mode == 'NONE':
            pad_node = 0
            pad_edge = 0
        else:
            pad_node = 0
            pad_edge = 0
        print(f'Max edges: {max_edges}, Max nodes: {max_nodes}, Padding to {pad_edge} edges and {pad_node} nodes')
        pad_start = time.time()
        if self.padding_mode == 'NODE':
            for i in range(len(self.graphs)):
                unbatched_g = dgl.unbatch(self.graphs[i])
                max_num_nodes = max(g.number_of_nodes() for g in unbatched_g)
                self.graphs[i] = utils.pad_batch_num_nodes(self.graphs[i], max_num_nodes, hid_size=self.hid_size)
                self.batch_num_nodes.append(self.graphs[i].batch_num_nodes())
                self.batch_num_edges.append(self.graphs[i].batch_num_edges())
        else:
            for i in range(len(self.graphs)):
                self.graphs[i] = utils.pad_batch(self.graphs[i], pad_edge, pad_node)
                self.batch_num_nodes.append(self.graphs[i].batch_num_nodes())
                self.batch_num_edges.append(self.graphs[i].batch_num_edges())
        pad_end = time.time()
        print(f'Padded {len(self.graphs)} batches in {pad_end - pad_start} seconds')

    def save(self):
        if not self.save_to_disk:
            return
        graph_path = os.path.join(self.save_dir, f'{self.name}_{self.chunkno}_{self.suffix}.bin')
        print(f'Saving dataset to {graph_path}')
        if len(self.graphs) == 0:
            return
        dgl.save_graphs(str(graph_path), self.graphs, {'labels': torch.stack(self.labels), 'batch_num_nodes': torch.stack(self.batch_num_nodes), 'batch_num_edges': torch.stack(self.batch_num_edges), 'tracking': torch.stack(self.tracking), 'globals': torch.stack(self.globals)})

    def has_cache(self):
        if not self.save_to_disk:
            return False
        for ch in range(self.chunks):
            graph_path = os.path.join(self.save_dir, f'{self.name}_{ch}_{self.suffix}.bin')
            if not os.path.exists(graph_path):
                print(f'Cache file {graph_path} does not exist, not loading from cache.')
                return False
        return True

    def load(self):
        if not self.save_to_disk:
            return
        self.graphs = []
        label_chunks = []
        tracking_chunks = []
        global_chunks = []
        for ch in range(self.chunks):
            graph_path = os.path.join(self.save_dir, f'{self.name}_{ch}_{self.suffix}.bin')
            print(f'Loading dataset from {graph_path}')
            graphs, label_dict = dgl.load_graphs(graph_path)
            label_chunks.append(label_dict['labels'])
            tracking_chunks.append(label_dict['tracking'])
            global_chunks.append(label_dict['globals'])
            for g, bnn, bne in zip(graphs, label_dict['batch_num_nodes'], label_dict['batch_num_edges']):
                g.set_batch_num_nodes(bnn)
                g.set_batch_num_edges(bne)
            self.graphs.extend(graphs)
        self.labels = torch.cat(label_chunks)
        self.tracking = torch.cat(tracking_chunks)
        self.globals = torch.cat(global_chunks)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.tracking[idx], self.globals[idx]
    
    def __len__(self):
        return len(self.graphs)

#Dataset which contains prebatched shuffled graphs. Cannot be saved to disk, else batching info is lost.
class LazyPreBatchedDataset(PreBatchedDataset):
    def __init__(self, **kwargs):
        # print(f'Unused kwargs: {kwargs}')
        self.current_chunk = None
        self.current_chunk_idx = -10
        self.label_chunks = []
        super().__init__(**kwargs)
    
    def load(self):
        if not self.save_to_disk:
            return
        self.label_chunks = []
        for ch in range(self.chunks):
            graph_path = os.path.join(self.save_dir, f'{self.name}_{ch}_{self.suffix}.bin')
            print(f'Loading dataset from {graph_path}')
            label_dict = dgl.data.graph_serialize.load_labels_v2(graph_path)
            self.label_chunks.append(label_dict)

    def __getitem__(self, idx):
        chunk_idx = -1
        sum = 0
        ev_idx = -999
        for i in range(len(self.label_chunks)):
            count = len(self.label_chunks[i]['labels'])
            if idx < sum + count:
                chunk_idx = i
                ev_idx = idx - sum
                break
            sum += count
        if chunk_idx != self.current_chunk_idx:
            # print(f"rank {self.rank} getting data from {self.name}_{chunk_idx}_{self.suffix}.bin")
            self.current_chunk, _ = dgl.load_graphs(os.path.join(self.save_dir, f'{self.name}_{chunk_idx}_{self.suffix}.bin'))
            self.current_chunk_idx = chunk_idx
        g = self.current_chunk[ev_idx]
        g.set_batch_num_nodes(self.label_chunks[chunk_idx]['batch_num_nodes'][ev_idx])
        g.set_batch_num_edges(self.label_chunks[chunk_idx]['batch_num_edges'][ev_idx])
        return g, self.label_chunks[chunk_idx]['labels'][ev_idx], self.label_chunks[chunk_idx]['tracking'][ev_idx], self.label_chunks[chunk_idx]['globals'][ev_idx]

    def __len__(self):
        l = 0
        for chunk in self.label_chunks:
            l += len(chunk['labels'])
        return l
