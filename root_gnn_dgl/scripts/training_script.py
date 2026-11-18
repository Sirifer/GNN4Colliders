import argparse
import time
import datetime
import yaml
import os
from pathlib import Path

start_time = time.time()

import dgl
import torch
import torch.nn as nn

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
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

print("import time: {:.4f} s".format(time.time() - start_time))

def mem():
    print(f'Current memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024} GB')

def gpu_mem():
    print()
    print('GPU Memory Usage:')
    sum = 0
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(obj.numel() if len(obj.size()) > 0 else 0, type(obj), obj.size())
    #             sum += obj.numel() if len(obj.size()) > 0 else 0
    #     except:
    #         pass
    print(f'Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB')
    # print(f'Current GPU cache usage: {torch.cuda.memory_cached() / 1024 / 1024 / 1024} GB')
    # print(f'Current GPU max memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB')
    # print(f'Current GPU max cache usage: {torch.cuda.max_memory_cached() / 1024 / 1024 / 1024} GB')
    # print(f'Numel in current tensors: {sum}')
    mem()


## epoch stores the epoch number I want to evaluate the model at
def evaluate(val_loaders, model, config, device, epoch = -1):
    print("Evaluating")
    
    if (epoch != -1) :
        print(f"Evalulating at epoch {epoch}")
        last_ep, checkpoint = utils.get_specific_epoch(config, epoch, from_ryan=False)
        print(f"Evaluating at epoch = {last_ep}")
    else:
        starting_epoch = 0
        last_ep, checkpoint = utils.get_last_epoch(config)   
        
    if checkpoint != None:
        ep = last_ep
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        starting_epoch = checkpoint['epoch'] + 1
        print(f"Loaded epoch {checkpoint['epoch']} from checkpoint")

    if 'Loss' not in config:
        loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_fcn = utils.buildFromConfig(config['Loss'], {'reduction': 'none'})
    if len(val_loaders) == 0:
        return "No validation data"
    start = time.time()
    scores = []
    labels = []
    weights = []
    before_decoder = []
    after_decoder = []
    tracking = []

    batch_size = config["Training"]["batch_size"]

    batch_limit = int(np.ceil(1e5 / batch_size))

    model.eval()
    with torch.no_grad():
        for loader in val_loaders:
            batch_count = 0
            for batch, label, track, global_feats in loader:
                #Don't use compiled model for testing since we can't control the batch size.
                #We could before, but it assumes each dataset has the same number of batches...
                before_global_decoder, after_global_decoder, after_classify = model.representation(batch.to(device), global_feats.to(device))
                
                scores.append(after_classify.to("cpu"))
                before_decoder.append(before_global_decoder.to("cpu"))
                after_decoder.append(after_global_decoder.to("cpu"))
                labels.append(label.to("cpu"))
                weights.append(track[:,1].to("cpu"))
                tracking.append(track.to("cpu"))

                batch_count += 1
                if batch_count >= batch_limit:
                    break

    if scores == []: #If validation set is empty.
        return
    logits = torch.concatenate(scores)
    scores = torch.sigmoid(logits)
    labels = torch.concatenate(labels)
    weights = torch.concatenate(weights)
    before_decoder = torch.concatenate(before_decoder)
    after_decoder = torch.concatenate(after_decoder)
    tracking = torch.concatenate(tracking)

    logits = logits.to("cpu").numpy()
    scores = scores.to("cpu").numpy()
    labels = labels.to("cpu").numpy()
    before_decoder = before_decoder.to("cpu").numpy()
    after_decoder = after_decoder.to("cpu").numpy()
    tracking = tracking.to("cpu").numpy()

    # Save the NumPy arrays to a .npz file
    outfile = f"{config['Training_Directory']}/evaluation_{epoch}.npz"

    np.savez(outfile, logits=logits, scores=scores, labels=labels, before_decoder=before_decoder, after_decoder=after_decoder, tracking=tracking)

    print(f"saved scores to {outfile}")
    return


def train(train_loaders, test_loaders, val_loaders, model, device, config, args, rank):
    nocompile = args.nocompile
    restart = args.restart
    # define train/val samples, loss function and optimizer
    if 'Loss' not in config:
        loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        finish_fn = torch.nn.Sigmoid()
    else:
        loss_fcn = utils.buildFromConfig(config['Loss'], {'reduction':'none'})
        finish_fn = utils.buildFromConfig(config['Loss'].get('finish', {'module': 'torch.nn', 'class': 'Sigmoid', 'args': {}}))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['Training']['learning_rate'])
    if 'gamma' in config['Training']:
        gamma = config['Training']['gamma']
    else:
        gamma = 1

    if 'dynamic_lr' in config['Training']:
        factor = config['Training']['dynamic_lr']['factor']
        patience = config['Training']['dynamic_lr']['patience']
    else:
        factor = 1
        patience = 1

    early_termination = utils.EarlyStop()
    if 'early_termination' in config['Training']:
        early_termination.patience = config['Training']['early_termination']['patience']
        early_termination.threshold = config['Training']['early_termination']['threshold']
        early_termination.mode = config['Training']['early_termination']['mode']

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma) 
    #scheduler_reset =  custom_scheduler.Dynamic_LR(optimizer, 'max', factor = factor, patience = patience)
    custom_scheduler = None
    if ('custom_scheduler' in config['Training']):
        run_time_args = {}
        scheduler_class = config['Training']['custom_scheduler']['class']
        if (scheduler_class == 'Dynamic_LR' or
                scheduler_class == 'Dynamic_LR_AND_Partial_Reset' or 
                    scheduler_class == 'Dynamic_LR_AND_Full_Reset'):
            
            run_time_args={'optimizer': optimizer}
        
        custom_scheduler = utils.buildFromConfig(config['Training']['custom_scheduler'], run_time_args=run_time_args)

    starting_epoch = 0
    if not restart:
        last_ep, checkpoint = utils.get_last_epoch(config)   
        if checkpoint != None:
            ep = starting_epoch - 1
            if nocompile:
                new_state_dict = {}
                for k, v in checkpoint['model_state_dict'].items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                checkpoint['model_state_dict'] = new_state_dict
                if (args.multinode or args.multigpu):
                    new_state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        new_key = 'module.' + k
                        new_state_dict[new_key] = v
                    checkpoint['model_state_dict'] = new_state_dict
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            if 'early_stop' in checkpoint:
                early_termination = utils.EarlyStop.load_from_dict(checkpoint['early_stop'])
                print(early_termination.to_str())
                print("EarlyStop state restored successfully.")
                if early_termination.should_stop:
                    print(f"Early Termination at Epoch {epoch}")
                    return
            else:
                print("'early_stop' not found in checkpoint. Initializing a new EarlyStop instance.")
                early_termination = utils.EarlyStop()
            print(f"Loaded epoch {checkpoint['epoch']} from checkpoint")
        log = open(config['Training_Directory'] + '/training.log', 'a', buffering=1)
    else:
        log = open(config['Training_Directory'] + '/training.log', 'w', buffering=1)

    train_cyclers = []
    for loader in train_loaders:
        train_cyclers.append(utils.cycler((loader)))

    if args.savecache:
        max_batch = [None,] * len(train_loaders)
        for dset_i, loader in enumerate(train_loaders):
            mbs = 0
            for batch_i, batch in enumerate(loader):
                if batch[0].num_nodes() > mbs:
                    mbs = batch[0].num_nodes()
                    max_batch[dset_i] = batch[0]
                    print(f'Max batch size for dataset {dset_i}: {mbs}')
        big_batch = dgl.batch(max_batch).to(device)
        with torch.no_grad():
            model(big_batch)

    cumulative_times = [0,0,0,0,0]
    log.write(f'Training {config["Training_Name"]} {datetime.datetime.now()} \n')
    print(f"Starting training for {config['Training']['epochs']} epochs")

    if hasattr(train_loaders[0].dataset, 'padding_mode'):
        is_padded = train_loaders[0].dataset.padding_mode != 'NONE'
        if (train_loaders[0].dataset.padding_mode == 'NODE'):
            is_padded = False
    else:
        is_padded = False

    lr_utils.print_LR(optimizer)

    # torch.save({
    #             'epoch': 0,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             }, os.path.join(config['Training_Directory'], f"model_epoch_{0}.pt"))
    # exit()


    # training loop
    # gpu_mem()
    for epoch in range(starting_epoch, config['Training']['epochs']):
        start = time.time()
        run = start
        if (args.profile):
            if (epoch == 0):
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("Epoch Start")

        if (args.multigpu or args.multinode):
            dist.barrier()
        
        if (epoch == 5):
            exit

        # training
        model.train()
        ibatch = 0
        total_loss = 0
        for batched_graph, labels, _, global_feats in train_loaders[0]:
            # # need to fix padded case
            # if is_padded:
            #     tglobals.append(torch.zeros(1, len(global_feats[0])))

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
            # loss = loss_fcn(logits, tlabels.to(device)) # changed logits from logits[:,0] and left labels as int for multiclass. Does this break binary? Yes.
            # loss = torch.sum(weights * loss) / torch.sum(weights)


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
            cumulative_times[0] += batch_start - run
            cumulative_times[1] += load - batch_start
            run = time.time()
            cumulative_times[2] += run - load
            if ibatch % 1000 == 0:
                print(f'Batch {ibatch} out of {len(train_loaders[0])}', end='\r')
                # gpu_mem()
        
        if (args.multigpu): 
            print(f'Rank {rank} Epoch Done.')
        elif (args.multinode): 
            print(f'Rank {args.global_rank} Epoch Done.')
        else:
            print("Epoch Done.")
        # validation

        scores = []
        labels = []
        weights = []
        model.eval()

        # Use the validation parity for monitoring / early stopping when available.
        # The held-out test parity is then never touched during training.
        eval_loaders = val_loaders if len(val_loaders) > 0 else test_loaders
        has_validation = len(val_loaders) > 0

        if (args.profile):
            torch.cuda.nvtx.range_push("Model Evaluation")

        with torch.no_grad():
            for loader in eval_loaders:
                for batch, label, track, global_feats in loader:
                    #Don't use compiled model for testing since we can't control the batch size.
                    #We could before, but it assumes each dataset has the same number of batches...
                    if is_padded:
                        global_feats = torch.cat([global_feats, torch.zeros(1, len(global_feats[0]))])
                    if nocompile:
                        batch_scores = model(batch.to(device), global_feats.to(device))
                    else:
                        batch_scores = model._orig_mod(batch.to(device), global_feats.to(device))
                    if is_padded:
                        scores.append(batch_scores[:-1,:])
                    else:
                        scores.append(batch_scores)
                    labels.append(label)
                    weights.append(track[:,1])
        eval_end = time.time()
        cumulative_times[3] += eval_end - run

        if (args.profile):
            torch.cuda.nvtx.range_pop() # pop evaluation

        if scores == []: #If validation set is empty.
            continue
        logits = torch.concatenate(scores).to(device)
        labels = torch.concatenate(labels).to(device)
        weights = torch.concatenate(weights).to(device)

        if (args.multigpu or args.multinode):
            gathered_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
            gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            gathered_weights = [torch.zeros_like(weights) for _ in range(dist.get_world_size())]

        if (args.multigpu or args.multinode):
            dist.barrier()
            if (args.multigpu and rank != 0) or (args.multinode and args.global_rank != 0):
                dist.gather(logits, dst=0)
                dist.gather(labels, dst=0)
                dist.gather(weights, dst=0)
                continue
            else:
                dist.gather(logits, gather_list=gathered_logits)
                dist.gather(labels, gather_list=gathered_labels)
                dist.gather(weights, gather_list=gathered_weights)
                
            logits = torch.concatenate(gathered_logits)
            labels = torch.concatenate(gathered_labels)
            weights = torch.concatenate(gathered_weights)

        wgt_mask = weights > 0

        if args.abs:
            weights = torch.abs(weights)

        print(f"Num batches trained = {ibatch}")

        #Note: This section is a bit ugly. Very conditional. Should maybe config defined behavior?
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
            threshold = 0.1  # 10% threshold

            for i in range(len(labels[0])):
                # accurate_count = torch.sum(torch.abs(preds[:, i].to("cpu") - labels[:, i].to("cpu")) / labels[:, i].to("cpu") <= threshold)
                # multilabel_accruacy.append(accurate_count / len(labels))
                multilabel_accuracy.append(torch.sum(preds[:, i].to("cpu") == labels[:, i].to("cpu")) / len(labels))
            test_auc = 0
            acc = np.mean(multilabel_accuracy)

        elif logits.shape[1] == 1 and loss_fcn.__class__.__name__ == 'BCEWithLogitsLoss': #Proxy for binary classification.
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
        

        # print(f"logits = {logits[:10]}")
        # print(f"preds = {preds[:2]}")
        # print(f"labels = {labels[:10]}")

        # print(f"len(Unique logits) = {len(torch.unique(logits))}")
        # print(f"Average of labels = {torch.mean(labels)}")
        # print(f"unique logits = {torch.unique(logits)[0]:.4f}, {torch.unique(logits)[-1]:.4f}")


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

        # test_loss = loss_fcn(logits, labels.to(device))
        # test_loss = loss_fcn(logits, labels)
        # test_loss = torch.sum(weights * test_loss) / torch.sum(weights)

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
        acc_name = "Val_Accuracy" if has_validation else "Accuracy"
        loss_name = "Val_Loss" if has_validation else "Test_Loss"
        auc_name = "Val_AUC" if has_validation else "Test_AUC"
        log_str = (
            f"Epoch {epoch:05d} | LR {optimizer.param_groups[0]['lr']:.4e} | "
            f"Loss {total_loss/ibatch:.4f} | {acc_name} {acc:.4f} | "
            f"{loss_name} {test_loss:.4f} | {auc_name} {test_auc:.4f} | "
            f"Time {end - start:.4f} s"
        )
        log.write(log_str + '\n')
        print(log_str, flush=True)

        state_dict = model.state_dict()         
        if not nocompile:
            state_dict = model._orig_mod.state_dict()

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        # print('Testing done')
        # gpu_mem()

        if epoch == 2:
            # torch.cuda.cudart().cudaProfilerStop()
            pass

        torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'early_stop': early_termination.to_dict()
                }, os.path.join(config['Training_Directory'], f"model_epoch_{epoch}.pt"))
        np.savez(os.path.join(config['Training_Directory'], f'model_epoch_{epoch}.npz'), scores=scores.to("cpu"), labels=labels.to("cpu"))
        save_end = time.time()
        cumulative_times[4] += save_end - eval_end

        early_termination.update(test_loss)
        if early_termination.should_stop:
            log_str = f"Early Termination at Epoch {epoch}"
            log.write(log_str + "\n")
            print(log_str)
            log_str = early_termination.to_str()
            log.write(log_str + "\n")
            print(log_str)
            break

        if (custom_scheduler):
            custom_scheduler.step(model, {'test_auc':test_auc})
        scheduler.step()

        if (args.profile):
            torch.cuda.nvtx.range_pop() # pop epoch

    print(f"Load: {cumulative_times[0]:.4f} s")
    print(f"Batch: {cumulative_times[1]:.4f} s")
    print(f"Train: {cumulative_times[2]:.4f} s")
    print(f"Eval: {cumulative_times[3]:.4f} s")
    print(f"Save: {cumulative_times[4]:.4f} s")
    log.close()

def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def init_process_group(world_size, rank, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = find_free_port()
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs (other was gloo)
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300),
    )

def main(rank=0, args=None, world_size=1, port=24500, seed=12345):

    #Prevent simultaneous file access
    #sleep_time = 120 * rank
    #time.sleep(sleep_time)

    #Load config file
    config = utils.load_config(args.config)

    if (args.directory):
        print(f"New training directory: { config['Training_Directory'] + args.directory}")
        config['Training_Directory'] = config['Training_Directory'] + args.directory

    if not os.path.exists(config['Training_Directory']):
        os.makedirs(config['Training_Directory'], exist_ok=True)
    with open(config['Training_Directory'] + '/config.yaml', 'w') as f:
        yaml.dump(config, f)
    batch_size = config["Training"]["batch_size"]

    if(args.plot):
        rl = utils.read_log(config)
        utils.plot_log(rl, config['Training_Directory'] + '/training.png')
        print('Log at ' + config['Training_Directory'] + '/training.log')
        print('Plotted at ' + config['Training_Directory'] + '/training.png')
        exit()
    
    if (args.multigpu):
        print(f"Setting up multigpu")
        start_time = time.time()
        init_process_group(world_size, rank, port)
        print("multigpu setup time: {:.4f} s".format(time.time() - start_time))
        device = torch.device(f'cuda:{rank}')
        torch.cuda.device(device)
    elif (args.multinode):
        device = torch.device(f'cuda:{rank}')
        torch.cuda.device(device)
        print(f"global rank = {args.global_rank}, local rank = {rank}, device = {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (args.cpu):
        print(f"Using CPU")
        device = "cpu"

    train_loaders = []
    test_loaders = []
    val_loaders = []
    load_start = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True

    ldr_type = datasets.LazyPreBatchedDataset if args.lazy else datasets.PreBatchedDataset

    #Load datasets
    if (pargs.statistics):
        pargs.statistics = int(pargs.statistics)
        print(f"Training Dataset Size: {pargs.statistics}")
        num_batches = int(np.ceil(pargs.statistics / batch_size))
        np.random.seed(pargs.seed)

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
            ldr = ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=mask_fn, suffix = utils.fold_selection_name(fold_conf, 'train'), chunks = shuffle_chunks, padding_mode = padding_mode, hidden_size = config["Model"]["args"]["hid_size"])
            gsamp, _, _, global_samp = ldr[0]
            sampler = None
            
            if (pargs.statistics):
                sampler = np.random.choice(range(len(ldr)), size=num_batches)

            if (args.multigpu):
                sampler = DistributedSampler(ldr, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
                # num_batches = len(ldr)
                # sampler = list(sampler)
                # if (sampler[0] >= num_batches % world_size):
                #     sampler.pop()
            if (args.multinode):
                sampler = DistributedSampler(ldr, num_replicas=world_size, rank=pargs.global_rank, shuffle=False, drop_last=True)
            if args.epoch_shuffle and sampler is None:
                train_loaders.append(torch.utils.data.DataLoader(ldr, batch_size=None, num_workers=0, shuffle=True))
            else:
                train_loaders.append(torch.utils.data.DataLoader(ldr, batch_size = None, num_workers = 0, sampler = sampler))
            sampler = None
            ldr = ldr_type(start_dataset=dset, batch_size=batch_size, mask_fn=mask_fn, suffix = utils.fold_selection_name(fold_conf, 'test'), chunks = shuffle_chunks, padding_mode = padding_mode, hidden_size= config['Model']['args']['hid_size'])
            if (args.multigpu):
                sampler = DistributedSampler(ldr, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
                # num_batches = len(ldr)
                # sampler = list(sampler)
                # if (rank >= num_batches % world_size):
                #     sampler.pop()
            if (args.multinode):
                sampler = DistributedSampler(ldr, num_replicas=world_size, rank=pargs.global_rank, shuffle=False, drop_last=True)

            test_loaders.append(torch.utils.data.DataLoader(ldr, batch_size = None, num_workers = 0, sampler=sampler))

            if "validation" in fold_conf:
                sampler = None
                ldr = ldr_type(start_dataset=dset, batch_size=batch_size,
                               mask_fn=utils.fold_selection(fold_conf, "validation"),
                               suffix=utils.fold_selection_name(fold_conf, "validation"),
                               chunks=shuffle_chunks, padding_mode=padding_mode,
                               hidden_size=config['Model']['args']['hid_size'])
                val_loaders.append(torch.utils.data.DataLoader(
                    ldr, batch_size=None, num_workers=0, sampler=sampler))
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
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters = {pytorch_total_params}")
    if not args.nocompile:
        model = torch.compile(model)
    if args.multigpu:
        print(f"Trying to create DDP model")
        start_time = time.time()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        print("model creation time: {:.4f} s".format(time.time() - start_time))
    if (args.multinode):
        print(f"Trying to create DDP model")
        start_time = time.time()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        print("model creation time: {:.4f} s".format(time.time() - start_time))

    # total_params = 0
    # for param_dict in model.parameters():
    #     for param in param_dict['params']:
    #         if param.requires_grad:
    #             total_params += param.numel()
    # print(f"Number of trainable parameters = {total_params}")

    if(type(model) == GCN.Clustering):
        print("clustering")

    if args.evaluate != None:
        evaluate(test_loaders, model, config, device, args.evaluate)
        exit()

    # model training
    print("Training...")
    gpu_mem()
    train(train_loaders, test_loaders, val_loaders, model, device, config, args, rank)

    # test the model
    # print("Testing...")
    # evaluate(val_loaders, model, config, device)

    # if args.multigpu or args.multinode:
    #     dist.destroy_process_group()

    # if rank == 0:
    #     rl = utils.read_log(config)
    #     utils.plot_log(rl, config['Training_Directory'] + '/training.png')
    #     print('Log at ' + config['Training_Directory'] + '/training.log')
    #     print('Plotted at ' + config['Training_Directory'] + '/training.png')

if __name__ == "__main__":
    #Handle CLI arguments
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("--config", type=str, help="Config file.", required=True)
    add_arg("--restart", action="store_true", help="Restart training from scratch.")
    add_arg("--preshuffle", action="store_true", help="Shuffle data before training.")
    add_arg("--epoch-shuffle", action="store_true",
            help="With --preshuffle, reshuffle the prebatched batches every epoch.")
    add_arg("--lazy", action="store_true", help="Lazy loading of data.")
    add_arg("--nocompile", action="store_true", help="Disable JIT compilation.")
    add_arg("--evaluate", type = int, help="Skip training and go to evaluation.")
    add_arg("--plot", action="store_true", help="Plot training logs.")
    add_arg("--multigpu", action="store_true", help="Use multiple GPUs.")
    add_arg("--multinode", action="store_true", help="Use multiple nodes.")
    add_arg("--savecache", action="store_true", help="")
    add_arg("--cpu", action="store_true", help="Uses the cpu only")
    add_arg("--statistics", type=float, help="Size of training data")
    add_arg("--directory", type=str, help="Append to Training Directory")
    add_arg("--seed", type=int, default=2, help="Sets random seed")
    add_arg("--abs", action="store_true", help="Use abs value of per-event weight")
    add_arg("--profile", action="store_true", help="use nsight systems profiler")

    pargs = parser.parse_args()
    
    if pargs.multigpu:
        port = find_free_port()
        torch.backends.cudnn.enabled = False
        mp.spawn(main, args=(pargs, 4, port), nprocs=4, join=True)
    if pargs.multinode:
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"global_rank = {global_rank}, local_rank = {local_rank}, world_size = {world_size}")
        
        dist.init_process_group(backend="nccl")
        torch.backends.cudnn.enabled = False

        pargs.global_rank = global_rank
        
        main(rank = local_rank, args=pargs, world_size=world_size)
    else:
        main(0, pargs)
        
        
