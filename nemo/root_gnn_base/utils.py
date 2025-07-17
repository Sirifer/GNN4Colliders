import importlib
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import dgl
import signal

def buildFromConfig(conf, run_time_args = {}):
    device = run_time_args.get('device', 'cpu')
    if 'module' in conf:
        module = importlib.import_module(conf['module'])
        cls = getattr(module, conf['class'])
        args = conf['args'].copy()
        if 'weight' in args and isinstance(args['weight'], list):
            args['weight'] = torch.tensor(args['weight'], dtype=torch.float, device=device)
        # Remove device from run_time_args to not pass it to the class
        run_time_args = {k: v for k, v in run_time_args.items() if k != 'device'}
        return cls(**args, **run_time_args)
    else:
        print('No module specified in config. Returning None.')

def cycler(iterable):
    while True:
        #print('Cycler is cycling...')
        for i in iterable:
            yield i

def include_config(conf):
    if 'include' in conf:
        for i in conf['include']:
            with open(i) as f:
                conf.update(yaml.load(f, Loader=yaml.FullLoader))
        del conf['include']

def load_config(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    include_config(conf)
    return conf

#Timeout function from https://stackoverflow.com/questions/492519/timeout-on-a-function-call
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def set_timeout(timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

def unset_timeout():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL) 

def make_padding_graph(batch, pad_nodes, pad_edges):
    senders = []
    receivers = []
    senders = torch.arange(0,pad_edges) // pad_nodes
    receivers = torch.arange(1,pad_edges+1) % pad_nodes
    if pad_nodes < 0 or pad_edges < 0 or pad_edges > pad_nodes * pad_nodes / 2:
        print('Batch is larger than padding size or e > n^2/2. Repeating edges as necessary.')
        print(f'Batch nodes: {batch.num_nodes()}, Batch edges: {batch.num_edges()}, Padding nodes: {pad_nodes}, Padding edges: {pad_edges}')
        senders = senders % pad_nodes
    padg = dgl.graph((senders[:pad_edges], receivers[:pad_edges]), num_nodes = pad_nodes)
    for k in batch.ndata.keys():
        padg.ndata[k] = torch.zeros( (pad_nodes, batch.ndata[k].shape[1]) )
    for k in batch.edata.keys():
        padg.edata[k] = torch.zeros( (pad_edges, batch.edata[k].shape[1]) )
    return dgl.batch([batch, padg.to(batch.device)])

def pad_size(graphs, edges, nodes, edge_per_graph=3, node_per_graph=14):
    pad_nodes = ((nodes // (node_per_graph * graphs))+1) * graphs * node_per_graph
    pad_edges = ((edges // (edge_per_graph * graphs))+1) * graphs * edge_per_graph
    return pad_nodes, pad_edges

def pad_batch_to_step_per_graph(batch, edge_per_graph=3, node_per_graph=14):
    n_graphs = batch.batch_num_nodes().shape[0]
    pad_nodes = (batch.num_nodes() + node_per_graph * n_graphs) % int(n_graphs * node_per_graph)
    pad_edges = (batch.num_edges() + edge_per_graph * n_graphs) % int(n_graphs * edge_per_graph)
    return make_padding_graph(batch, pad_nodes, pad_edges)

def pad_batch(batch, edges = 104000, nodes = 16000):
    if edges == 0 and nodes == 0:
        return batch
    pad_nodes = 0
    pad_edges = 0
    pad_nodes = nodes - batch.num_nodes()
    pad_edges = edges - batch.num_edges()
    return make_padding_graph(batch, pad_nodes, pad_edges)

def pad_batch_num_nodes(batch, max_num_nodes, hid_size = 64):
    print(f"Padding each graph to have {max_num_nodes} nodes. Using hidden size {hid_size}.")

    unbatched = dgl.unbatch(batch)
    for g in unbatched:
        num_nodes_to_add = max_num_nodes - g.number_of_nodes()
        if num_nodes_to_add > 0:
            g.add_nodes(num_nodes_to_add)  # Add isolated nodes

    batch = dgl.batch(unbatched)

    padding_mask = torch.zeros((batch.ndata['features'].shape[0]), dtype=torch.bool)
    global_update_weights = torch.ones((batch.ndata['features'].shape[0], hid_size))

    for i in range(len(batch.ndata['features'])):
        if (torch.count_nonzero(batch.ndata['features'][i]) == 0):
            padding_mask[i] = True
            global_update_weights[i] = 0

    batch.ndata['w'] = global_update_weights
    batch.ndata['padding_mask'] = padding_mask

    return batch


def fold_selection(fold_config, sample):
    n_folds = fold_config['n_folds']
    folds_opt = fold_config[sample]
    folds = []
    if type(folds_opt) == int:
        return lambda x : x.tracking[:,0] % n_folds == folds_opt
    elif type(folds_opt) == list:
        print("fold type is list")
        print(f"fold_config = {fold_config}")
        print(f"folds_opt = {folds_opt}")
        return lambda x : sum([x.tracking[:,0] % n_folds == f for f in folds_opt]) == 1
    else:
        raise ValueError("Invalid fold selection option with type {}".format(type(folds_opt)))
    
def fold_selection_name(fold_config, sample):
    n_folds = fold_config['n_folds']
    folds_opt = fold_config[sample]
    if type(folds_opt) == int:
        return f'n_{n_folds}_f_{folds_opt}'
    elif type(folds_opt) == list:
        return f'n_{n_folds}_f_{"_".join([str(f) for f in folds_opt])}'
    else:
        raise ValueError("Invalid fold selection option with type {}".format(type(folds_opt)))

#Return the index and checkpoint of the last epoch.
def get_last_epoch(config, max_ep = -1, device = None):
    last_epoch = -1
    checkpoint = None
    if max_ep < 0:
        max_ep = config['Training']['epochs']
    for ep in range(max_ep):
        if os.path.exists(os.path.join(config['Training_Directory'], f'model_epoch_{ep}.pt')):
            last_epoch = ep
        else:
            print(f'Epoch {ep} not found. Stopping at epoch {last_epoch}')
            print('File not found: ', os.path.join(config['Training_Directory'], f'model_epoch_{ep}.pt'))
            break
    if last_epoch >= 0:
        checkpoint = torch.load(os.path.join(config['Training_Directory'], f'model_epoch_{last_epoch}.pt'), map_location=device)
    return last_epoch, checkpoint

#Return the index and checkpoint of the last epoch.
def get_specific_epoch(config, target_epoch, device = None, from_ryan = False):
    last_epoch = -1
    checkpoint = None
    for ep in range(target_epoch + 1):
        if (from_ryan):
            if os.path.exists(os.path.join('/global/cfs/cdirs/atlas/berobert/root_gnn_dgl/' + config['Training_Directory'], f'model_epoch_{ep}.pt')):
                last_epoch = ep
            else:
                print(f'Epoch {ep} not found. Stopping at epoch {last_epoch}')
                print('File not found: ', os.path.join('/global/cfs/cdirs/atlas/berobert/root_gnn_dgl/' + config['Training_Directory'], f'model_epoch_{ep}.pt'))
                break
        else:
            if os.path.exists(os.path.join(config['Training_Directory'], f'model_epoch_{ep}.pt')):
                last_epoch = ep
            else:
                print(f'Epoch {ep} not found. Stopping at epoch {last_epoch}')
                print('File not found: ', os.path.join(config['Training_Directory'], f'model_epoch_{ep}.pt'))
                break
    if last_epoch >= 0:
        if (from_ryan):
            checkpoint = torch.load('/global/cfs/cdirs/atlas/berobert/root_gnn_dgl/' + os.path.join(config['Training_Directory'], f'model_epoch_{last_epoch}.pt'), map_location=device)
        else:
            checkpoint = torch.load(os.path.join(config['Training_Directory'], f'model_epoch_{last_epoch}.pt'), map_location=device)
    return last_epoch, checkpoint

#Return the index and checkpoint of the nest epoch.
def get_best_epoch(config, var='Test_AUC', mode='max', device=None, from_ryan=False):
    # Read the training log
    log = read_log(config)
    
    # Ensure the specified variable exists in the log
    if var not in log:
        raise ValueError(f"Variable '{var}' not found in the training log.")
    
    # Determine the target epoch based on the mode ('max' or 'min')
    if mode == 'max':
        target_epoch = int(np.argmax(log[var]))
        print(f"Best epoch based on '{var}' (max): {target_epoch} with value: {log[var][target_epoch]}")
    elif mode == 'min':
        target_epoch = int(np.argmin(log[var]))
        print(f"Best epoch based on '{var}' (min): {target_epoch} with value: {log[var][target_epoch]}")
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'max' or 'min'.")
    
    # Initialize checkpoint retrieval variables
    last_epoch = -1
    checkpoint = None

    # Iterate through epochs up to the target epoch to find the corresponding checkpoint
    for ep in range(target_epoch + 1):
        if from_ryan:
            checkpoint_path = os.path.join(
                '/global/cfs/cdirs/atlas/berobert/root_gnn_dgl/',
                config['Training_Directory'],
                f'model_epoch_{ep}.pt'
            )
        else:
            checkpoint_path = os.path.join(
                config['Training_Directory'],
                f'model_epoch_{ep}.pt'
            )
        
        if os.path.exists(checkpoint_path):
            last_epoch = ep
        else:
            print(f'Epoch {ep} not found. Stopping at epoch {last_epoch}')
            print('File not found: ', checkpoint_path)
            break

    # Load the checkpoint for the last valid epoch
    if last_epoch >= 0:
        if from_ryan:
            checkpoint_path = os.path.join(
                '/global/cfs/cdirs/atlas/berobert/root_gnn_dgl/',
                config['Training_Directory'],
                f'model_epoch_{last_epoch}.pt'
            )
        else:
            checkpoint_path = os.path.join(
                config['Training_Directory'],
                f'model_epoch_{last_epoch}.pt'
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return last_epoch, checkpoint

def read_log(config):
    lines = []
    with open(config['Training_Directory'] + '/training.log', 'r') as f:
        lines = f.readlines()
    lines = [l for l in lines if 'Epoch' in l]
    
    labels = []
    for field in lines[0].split('|'):
        labels.append(field.split()[0])
    
    # Initialize log as a dictionary with empty lists
    log = {label: [] for label in labels}
    
    for line in lines:
        valid_row = True  # Flag to check if the row is valid
        temp_row = {}  # Temporary row to store values before adding to log
        
        for field in line.split('|'):
            spl = field.split()
            try:
                temp_row[spl[0]] = float(spl[1])
            except (ValueError, IndexError):
                valid_row = False  # Mark row as invalid if conversion fails
                break
        
        if valid_row:  # Only add the row if all fields are valid
            for label in labels:
                log[label].append(temp_row.get(label, np.nan))  # Handle missing labels gracefully
    
    # Convert lists to numpy arrays for consistency
    for label in labels:
        log[label] = np.array(log[label])
    
    return log

#Plot training logs.
def plot_log(log, output_file):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    #Time

    ax[0][0].plot(log['Epoch'], np.cumsum(log['Time']), label='Time')
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Time (s)')
    ax[0][0].legend()

    """
    ax[0][0].plot(log['Epoch'], log['LR'], label='Learning Rate')
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Learning Rate')
    ax[0][0].set_yscale('log')
    ax[0][0].legend()
    """

    #Loss
    ax[0][1].plot(log['Epoch'], log['Loss'], label='Train Loss')
    ax[0][1].plot(log['Epoch'], log['Test_Loss'], label='Test Loss')
    ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel('Loss')
    ax[0][1].legend()

    #Accuracy
    ax[1][0].plot(log['Epoch'], log['Accuracy'], label='Test Accuracy')
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel('Accuracy')
    ax[1][0].set_ylim((0.44, 0.56))
    ax[1][0].legend()

    #AUC
    ax[1][1].plot(log['Epoch'], log['Test_AUC'], label='Test AUC')
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel('AUC')
    ax[1][1].legend()

    fig.savefig(output_file)

class EarlyStop():
    def __init__(self, patience=15, threshold=1e-8, mode='min'):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.count = 0
        self.current_best = np.inf if mode == 'min' else -np.inf
        self.should_stop = False

    def update(self, value):
        if self.mode == 'min':  # Minimizing loss
            if value < self.current_best - self.threshold:
                self.current_best = value
                self.count = 0
            else:
                self.count += 1
        elif self.mode == 'max':  # Maximizing metric
            if value > self.current_best + self.threshold:
                self.current_best = value
                self.count = 0
            else:
                self.count += 1

        # Check if patience is exceeded
        if self.count >= self.patience:
            self.should_stop = True

    def reset(self):
        self.count = 0
        self.current_best = np.inf if self.mode == 'min' else -np.inf
        self.should_stop = False

    def to_str(self):
        status = (
            f"EarlyStop Status:\n"
            f"  Mode: {'Minimize' if self.mode == 'min' else 'Maximize'}\n"
            f"  Patience: {self.patience}\n"
            f"  Threshold: {self.threshold:.3e}\n"
            f"  Current Best: {self.current_best:.6f}\n"
            f"  Consecutive Epochs Without Improvement: {self.count}\n"
            f"  Stopping Triggered: {'Yes' if self.should_stop else 'No'}"
        )
        return status

    def to_dict(self):

        return {
            'patience': self.patience,
            'threshold': self.threshold,
            'mode': self.mode,
            'count': self.count,
            'current_best': self.current_best,
            'should_stop': self.should_stop,
        }
    
    @classmethod
    def load_from_dict(cls, state_dict):
        instance = cls(
            patience=state_dict['patience'],
            threshold=state_dict['threshold'],
            mode=state_dict['mode']
        )
        instance.count = state_dict['count']
        instance.current_best = state_dict['current_best']
        instance.should_stop = state_dict['should_stop']
        return instance
    

def graph_augmentation(graph):
    print("Augmenting Graph")
    return