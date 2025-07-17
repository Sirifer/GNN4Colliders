import argparse
import time
import datetime
import yaml
import os
import glob

start_time = time.time()

import dgl
import torch
import torch.nn as nn

import sys
file_path = os.getcwd()
sys.path.append(file_path)
import root_gnn_base.batched_dataset as datasets
from root_gnn_base import utils
import root_gnn_base.custom_scheduler as lr_utils
from models import GCN

import numpy as np
from sklearn.metrics import roc_auc_score
import resource
import gc

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData
from dataclasses import dataclass

print("import time: {:.4f} s".format(time.time() - start_time))

def mem():
    print(f'Current memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} GB')

def gpu_mem():
    print()
    print('GPU Memory Usage:')
    print(f'Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB')
    print(f'Current GPU cache usage: {torch.cuda.memory_cached() / 1024 / 1024 / 1024:.2f} GB')
    print(f'Current GPU max memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB')
    print(f'Current GPU max cache usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.2f} GB')
    mem()

def train(train_loaders, test_loaders, model, device, config, args):
    restart = args.restart

    loss_fcn = utils.buildFromConfig(config['Loss'], {'reduction':'none'})
    finish_fn = utils.buildFromConfig(config['Loss']['finish'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['Training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['Training']['gamma']) 

    early_termination = utils.EarlyStop()
    if 'early_termination' in config['Training']:
        early_termination.patience = config['Training']['early_termination']['patience']
        early_termination.threshold = config['Training']['early_termination']['threshold']
        early_termination.mode = config['Training']['early_termination']['mode']

    starting_epoch = 0
    if not restart:
        last_ep, checkpoint = utils.get_last_epoch(config)   
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        early_termination = utils.EarlyStop.load_from_dict(checkpoint['early_stop'])
        if early_termination.should_stop:
            print(f"Early Termination at Epoch {last_ep}")
            return
        print(f"Loaded epoch {checkpoint['epoch']} from checkpoint")
        log = open(config['Training_Directory'] + '/training.log', 'a', buffering=1)
    else:
        # Remove all *.pt and *.npz files in the Training_Directory
        for pattern in ('*.pt', '*.npz'):
            for file in glob.glob(os.path.join(config['Training_Directory'], pattern)):
                os.remove(file)
        log = open(config['Training_Directory'] + '/training.log', 'w', buffering=1)

    train_cyclers = []
    for loader in train_loaders:
        train_cyclers.append(utils.cycler((loader)))

    log.write(f'Training {config["Training_Name"]} {datetime.datetime.now()} \n')
    print(f"Starting training for {config['Training']['epochs']} epochs")

    if hasattr(train_loaders[0].dataset, 'padding_mode'):
        is_padded = train_loaders[0].dataset.padding_mode != 'NONE'
        if (train_loaders[0].dataset.padding_mode == 'NODE'):
            is_padded = False
    else:
        is_padded = False

    lr_utils.print_LR(optimizer)

    # training loop
    for epoch in range(starting_epoch, config['Training']['epochs']):
        start = time.time()
        run = start
        if (args.profile):
            if (epoch == 0):
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("Epoch Start")

        # training

        model.train()

        ibatch = 0
        total_loss = 0
        for batched_graph, labels, _, global_feats in train_loaders[0]:
            batch_start = time.time()
            logits = torch.tensor([])
            tlabels = torch.tensor([])
            weights = torch.tensor([])
            batch_lengths = []
            for cycler in train_cyclers:
                graph, label, track, global_feats = next(cycler)
                graph = graph.to(device)
                label = label.to(device)
                track = track.to(device)
                global_feats = global_feats.to(device)
                if is_padded: #Padding the globals to match padded graphs.
                    global_feats = torch.concatenate((global_feats, torch.zeros(1, len(global_feats[0])).to(device)))
                load = time.time()
                if (args.profile):
                    torch.cuda.nvtx.range_push("Model Forward")
                if (len(logits) == 0):
                    logits = model(graph, global_feats)
                    tlabels = label
                    weights = track[:,1]
                else:
                    logits = torch.concatenate((logits, model(graph, global_feats)), dim=0)
                    tlabels = torch.concatenate((tlabels, label), dim=0)
                    weights = torch.concatenate((weights, track[:,1]), dim=0)
                batch_lengths.append(logits.shape[0] - 1)

                if (args.profile):
                    torch.cuda.nvtx.range_pop() # popping model forward

            if is_padded:
                keepmask = torch.full_like(logits[:,0], True, dtype=torch.bool)
                keepmask[batch_lengths] = False
                logits = logits[keepmask]
            tlabels = tlabels.to(torch.float)
            if logits.shape[1] == 1 and loss_fcn.__class__.__name__ == 'BCEWithLogitsLoss':
                logits = logits[:,0]
                tlabels = tlabels.to(torch.float)
            if loss_fcn.__class__.__name__ == 'CrossEntropyLoss':
                tlabels = tlabels.to(torch.long)

            if args.abs:
                weights = torch.abs(weights)

            loss = loss_fcn(logits, tlabels.to(device))
            # Normalize loss within each label
            unique_labels = torch.unique(tlabels)  # Get unique labels
            normalized_loss = 0.0
            for label in unique_labels:
                # Mask for samples belonging to the current label
                label_mask = (tlabels == label)
                # Extract weights and losses for the current label
                label_weights = weights[label_mask]
                label_losses = loss[label_mask]
                # Compute normalized loss for the current label
                label_loss = torch.sum(label_weights * label_losses) / torch.sum(label_weights)
                # Add to the total normalized loss
                normalized_loss += label_loss
            loss = normalized_loss / len(unique_labels)

            if (args.profile):
                torch.cuda.nvtx.range_push("Model Backward")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item()

            if (args.profile):
                torch.cuda.nvtx.range_pop() # pop model backward

            ibatch += 1

            if ibatch % 1000 == 0:
                print(f'Batch {ibatch} out of {len(train_loaders[0])}', end='\r')
                # gpu_mem()
        else:
            print("Epoch Done.")
       
       # validation

        scores = []
        labels = []
        weights = []
        model.eval()

        if (args.profile):
            torch.cuda.nvtx.range_push("Model Evaluation")

        with torch.no_grad():
            for loader in test_loaders:
                for batch, label, track, global_feats in loader:
                    #Don't use compiled model for testing since we can't control the batch size.
                    #We could before, but it assumes each dataset has the same number of batches...
                    if is_padded:
                        global_feats = torch.cat([global_feats, torch.zeros(1, len(global_feats[0]))])

                    batch_scores = model(batch.to(device), global_feats.to(device))
                    
                    if is_padded:
                        scores.append(batch_scores[:-1,:])
                    else:
                        scores.append(batch_scores)
                    labels.append(label)
                    weights.append(track[:,1])

        if (args.profile):
            torch.cuda.nvtx.range_pop() # pop evaluation

        if scores == []: #If validation set is empty.
            continue

        logits = torch.concatenate(scores).to(device)
        labels = torch.concatenate(labels).to(device)
        weights = torch.concatenate(weights).to(device)

        wgt_mask = weights > 0

        if args.abs:
            weights = torch.abs(weights)

        print(f"Num batches trained = {ibatch}")

        if (loss_fcn.__class__.__name__ == "ContrastiveClusterLoss"):
            scores = logits
            preds = scores
            accuracy = 0
            test_auc = 0
            acc = 0
            contrastive_cluster_loss = finish_fn(logits)

        elif (loss_fcn.__class__.__name__ == "MultiLabelLoss"):
            scores = finish_fn(logits)
            preds = torch.round(scores)
            multilabel_accuracy = []
            threshold = 0.1
            for i in range(len(labels[0])):
                multilabel_accuracy.append(torch.sum(preds[:, i].to("cpu") == labels[:, i].to("cpu")) / len(labels))
            test_auc = 0
            acc = np.mean(multilabel_accuracy)

        elif logits.shape[1] == 1 and loss_fcn.__class__.__name__ == 'BCEWithLogitsLoss':
            test_auc = 0
            acc = 0
            logits = logits[:,0]
            scores = finish_fn(logits)
            labels =labels.to(torch.float)
            preds = scores > 0.5
            test_auc = roc_auc_score(labels[wgt_mask].to("cpu") == 1, scores[wgt_mask].to("cpu"), sample_weight=weights[wgt_mask].to("cpu"))
            acc = torch.sum(preds.to("cpu") == labels.to("cpu")) / len(labels)

        elif logits.shape[1] == 1 and loss_fcn.__class__.__name__ == 'MSELoss':
            logits = logits[:,0]
            scores = finish_fn(logits)
            labels = labels.to(torch.float)
            acc = 0
            test_auc = 0

        else:
            preds = torch.argmax(logits, dim=1)
            scores = finish_fn(logits)
            if labels.dim() == 1: #Multi-class
                acc = torch.sum(preds.to("cpu") == labels.to("cpu")) / len(labels) #TODO: Make each class weighted equally?
                
                labels = labels.to("cpu")
                weights = weights.to("cpu")
                logits = logits.to("cpu")
                wgt_mask = wgt_mask.to("cpu")

                labels_onehot = np.zeros((len(labels), len(scores[0])))
                labels_onehot[np.arange(len(labels)), labels] = 1

                try:
                    #test_auc = roc_auc_score(labels[wgt_mask].to("cpu") == 1, scores[wgt_mask].to("cpu"), multi_class='ovr', sample_weight=weights[wgt_mask].to("cpu"))
                    if (len(scores[0]) != config["Model"]["args"]["out_size"]):
                        print("ERROR: The out_size and the number of class labels don't match! Please check config.")
                    test_auc = roc_auc_score(labels_onehot[wgt_mask], scores[wgt_mask].to("cpu"), multi_class='ovr', sample_weight=weights[wgt_mask].to("cpu"))
                except ValueError:
                    test_auc = np.nan
            else: #Multi-loss
                acc = torch.sum(preds.to("cpu") == labels[:,0].to("cpu")) / len(labels)
                try:
                    test_auc = roc_auc_score(labels[:,0][wgt_mask].to("cpu") == 1, scores[wgt_mask].to("cpu"), multi_class='ovr', sample_weight=weights[wgt_mask].to("cpu"))
                except ValueError:
                    test_auc = np.nan

        if (loss_fcn.__class__.__name__ == "MultiLabelLoss"):
            multilabel_log_str = "MultiLabel_Accuracy "
            for accuracy in multilabel_accuracy:
                multilabel_log_str += f" | {accuracy:.4f}"
            log.write(multilabel_log_str + '\n')
            print(multilabel_log_str, flush=True)
        elif (loss_fcn.__class__.__name__ == "ContrastiveClusterLoss"):
            contrastive_cluster_log_str = "ContrastiveClusterLoss "
            contrastive_cluster_log_str += f"Contrastive Loss: {contrastive_cluster_loss[0]:.4f}, Clustering Loss: {contrastive_cluster_loss[1]:.4f}, Variance Loss: {contrastive_cluster_loss[2]:.4f}"
            log.write(contrastive_cluster_log_str + '\n')
            print(contrastive_cluster_log_str, flush=True)

        test_loss = loss_fcn(logits, labels)
        # Normalize loss within each label
        unique_labels = torch.unique(labels)  # Get unique labels
        normalized_loss = 0.0
        for label in unique_labels:
            # Mask for samples belonging to the current label
            label_mask = (labels == label)
            # Extract weights and losses for the current label
            label_weights = weights[label_mask]
            label_losses = test_loss[label_mask]
            # Compute normalized loss for the current label
            label_loss = torch.sum(label_weights * label_losses) / torch.sum(label_weights)
            # Add to the total normalized loss
            normalized_loss += label_loss
        test_loss = normalized_loss / len(unique_labels)


        end = time.time()
        log_str = "Epoch {:05d} | LR {:.4e} | Loss {:.4f} | Accuracy {:.4f} | Test_Loss {:.4f} | Test_AUC {:.4f} | Time {:.4f} s".format(
                epoch, optimizer.param_groups[0]['lr'], total_loss/ibatch, acc, test_loss, test_auc, end - start
        )
        log.write(log_str + '\n')
        print(log_str, flush=True)

        state_dict = model.state_dict()         

        torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'early_stop': early_termination.to_dict()
                }, os.path.join(config['Training_Directory'], f"model_epoch_{epoch}.pt"))
        np.savez(os.path.join(config['Training_Directory'], f'model_epoch_{epoch}.npz'), scores=scores.to("cpu"), labels=labels.to("cpu"))

        early_termination.update(test_loss)
        if early_termination.should_stop:
            log_str = f"Early Termination at Epoch {epoch}"
            log.write(log_str + "\n")
            print(log_str)
            log_str = early_termination.to_str()
            log.write(log_str + "\n")
            print(log_str)
            break

        scheduler.step()

        if (args.profile):
            torch.cuda.nvtx.range_pop() # pop epoch
            
    log.close()

def main(args=None):
    config = utils.load_config(args.config)

    if not os.path.exists(config['Training_Directory']):
        os.makedirs(config['Training_Directory'], exist_ok=True)
    with open(config['Training_Directory'] + '/config.yaml', 'w') as f:
        yaml.dump(config, f)
    batch_size = config["Training"]["batch_size"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders = []
    test_loaders = []
    val_loaders = []
    load_start = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True

    # ldr_type = datasets.LazyPreBatchedDataset if args.lazy else datasets.PreBatchedDataset
    ldr_type = datasets.LazyPreBatchedDataset

    for dset_conf in config["Datasets"]:
        dset = utils.buildFromConfig(config["Datasets"][dset_conf])
        if 'batch_size' in config["Datasets"][dset_conf]:
            batch_size = config["Datasets"][dset_conf]['batch_size']
        fold_conf = config["Datasets"][dset_conf]["folding"]
        shuffle_chunks = config["Datasets"][dset_conf].get("shuffle_chunks", 10)
        padding_mode = config["Datasets"][dset_conf].get("padding_mode", "STEPS")
        mask_fn = utils.fold_selection(fold_conf, "train")
        if args.preshuffle:
            # ldr = ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=mask_fn, suffix = utils.fold_selection_name(fold_conf, 'train'), chunks = shuffle_chunks, padding_mode = padding_mode, use_ddp = args.multigpu, rank=rank, world_size=world_size)
            ldr = ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=mask_fn, suffix = utils.fold_selection_name(fold_conf, 'train'), chunks = shuffle_chunks, padding_mode = padding_mode, hidden_size = 128)
            gsamp, _, _, global_samp = ldr[0]
            sampler = None
            train_loaders.append(torch.utils.data.DataLoader(ldr, batch_size = None, num_workers = 0, sampler = sampler))
            
            sampler = None
            ldr = ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=mask_fn, suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, padding_mode = padding_mode, hidden_size=128)
            test_loaders.append(torch.utils.data.DataLoader(ldr, batch_size = None, num_workers = 0, sampler=sampler))

            if "validation" in fold_conf:
                val_loaders.append(torch.utils.data.DataLoader((ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=utils.fold_selection(fold_conf, "validation"), suffix = utils.fold_selection_name(fold_conf, 'validation'), chunks = shuffle_chunks, hid_size=128,  padding_mode = padding_mode)), batch_size = None, num_workers = 0, sampler = sampler))
            else:
                print("No validation set for dataset ", dset_conf)
        else:
            train_loaders.append(datasets.GetBatchedLoader(dset, batch_size, utils.fold_selection(fold_conf, "train")))  
            gsamp, _, _, global_samp = dset[0]
            test_loaders.append(datasets.GetBatchedLoader(dset, batch_size, utils.fold_selection(fold_conf, "test")))
            if "validation" in fold_conf:
                val_loaders.append(datasets.GetBatchedLoader(dset, batch_size, utils.fold_selection(fold_conf, "validation")))
            else:
                print("No validation set for dataset ", dset_conf)
    
    load_end = time.time()
    print("Load time: {:.4f} s".format(load_end - load_start))

    model = utils.buildFromConfig(config["Model"], {'sample_graph': gsamp, 'sample_global': global_samp, 'seed': seed}).to(device)
    
    # @dataclass
    # class MetaData(ModelMetaData):
    #     name: str = "Edge_Network"

    physicsnemo_model = Module.from_torch(model)
    print(f"physicsnemo_model = {physicsnemo_model}")
    model = physicsnemo_model()
    

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters = {pytorch_total_params}")

    # model training
    print("Training...")
    gpu_mem()
    train(train_loaders, test_loaders, model, device, config, args)

if __name__ == "__main__":
    #Handle CLI arguments
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("--config", type=str, help="Config file.", required=True)
    add_arg("--restart", action="store_true", help="Restart training from scratch.")
    add_arg("--preshuffle", action="store_true", help="Shuffle data before training.")
    add_arg("--seed", type=int, default=2, help="Sets random seed")
    add_arg("--abs", action="store_true", help="Use abs value of per-event weight")
    add_arg("--profile", action="store_true", help="use nsight systems profiler")

    pargs = parser.parse_args()
    
    main(pargs)
        
        
