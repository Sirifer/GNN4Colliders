import time, os

start = time.time()
import torch
from torch.nn.parallel import DistributedDataParallel
from dgl.dataloading import GraphDataLoader
from torch.amp import GradScaler
import numpy as np
import hydra
from omegaconf import DictConfig
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.distributed.manager import DistributedManager

import json
from tqdm import tqdm
import random

import models.MeshGraphNet as MeshGraphNet
from dataset.Dataset import get_dataset
import metrics


class MGNTrainer:
    def __init__(self, logger, cfg, dist):
        # set device
        self.device = dist.device
        logger.info(f"Using {self.device} device")

        params = {}

        start = time.time()
        self.trainloader, self.valloader, self.testloader = get_dataset(cfg, self.device)
        print(f"total time loading dataset: {time.time() - start:.2f} seconds")

        dtype_str = getattr(cfg.root_dataset, "dtype", "torch.float32")
        if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
            self.dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
        else:
            self.dtype = torch.float32

        node_features = list(cfg.root_dataset.features.values())[0]
        edge_features = ["dR", "deta", "dphi"]
        global_features = ["num_nodes"]

        params["infeat_nodes"] = len(node_features)
        params["infeat_edges"] = len(edge_features)
        params["infeat_globals"] = len(global_features)
        params["out_dim"] = cfg.architecture.out_dim
        params["node_features"] = list(node_features)
        params["edge_features"] = edge_features
        params["global_features"] = global_features

        self.model = MeshGraphNet.MeshGraphNet(cfg.architecture)
        self.model = self.model.to(dtype=self.dtype, device=self.device)

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.scheduler.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.lr * cfg.scheduler.lr_decay,
        )
        self.scaler = GradScaler('cuda')

        # load checkpoint
        self.epoch_init = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.params = params
        self.cfg = cfg

    def backward(self, loss):
        """
        Perform backward pass.

        Arguments:
            loss: loss value.

        """
        # backward pass
        if self.cfg.performance.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def train(self, graph, metadata):
        """
        Perform one training iteration over one graph. The training is performed
        over multiple timesteps, where the number of timesteps is specified in
        the 'stride' parameter.

        Arguments:
            graph: the desired graph.

        Returns:
            loss: loss value.

        """
        graph = graph.to(self.device, non_blocking=True)
        globals = metadata['globals'].to(self.device, non_blocking=True)
        label = metadata['label'].to(self.device, non_blocking=True)
        weight =  metadata['weight'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        pred = self.model(graph.ndata["features"], graph.edata["features"], globals, graph, metadata)
        loss = metrics.weighted_bce(pred, label, weights=weight)
        self.backward(loss)
        return loss.detach()

    @torch.no_grad()
    def eval(self):
        """
        Evaluate the model on one batch.

        Args:
            graph (DGLGraph): The input graph.
            label (Tensor): The target labels.

        Returns:
            loss (Tensor): The computed loss value (scalar).
        """
        predictions = []
        labels = []
        weights = []

        for graph, metadata in self.valloader:
            
            graph = graph.to(self.device, non_blocking=True)
            globals = metadata['globals'].to(self.device, non_blocking=True)
            label = metadata['label'].to(self.device, non_blocking=True)
            weight =  metadata['weight'].to(self.device, non_blocking=True)
            
            pred = self.model(graph.ndata["features"], graph.edata["features"], globals, graph, metadata)
            predictions.append(pred)
            labels.append(label)
            weights.append(weight)     

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0)

        loss = metrics.weighted_bce(predictions, labels, weights=weights)
        
        # Convert logits to probabilities
        prob = torch.sigmoid(predictions)

        # Flatten to 1D arrays
        prob_flat = prob.detach().to(torch.float32).cpu().numpy().flatten()
        labels_flat = labels.detach().to(torch.float32).cpu().numpy().flatten()

        # Calculate AUC
        try:
            auc = metrics.roc_auc_score(labels_flat, prob_flat)
        except ValueError:
            auc = float('nan')  # Not enough classes present for AUC

        return loss, auc

@hydra.main(version_base=None, config_path="./configs/", config_name="tHjb_CP_0_vs_45")
def do_training(cfg: DictConfig):
    """
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    os.makedirs(cfg.checkpoints.ckpt_path, exist_ok=True)
    logger = PythonLogger("main")
    logger.file_logging(os.path.join(cfg.checkpoints.ckpt_path, "train.log"))

    # initialize trainer
    trainer = MGNTrainer(logger, cfg, dist)

    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            trainer.model = DistributedDataParallel(
                trainer.model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # training loop
    start = time.time()
    logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.training.epochs):

        # Training
        train_loss = []
        for graph, metadata in tqdm(trainer.trainloader, desc=f"epoch {epoch} trianing"):
            trainer.model.train()
            loss = trainer.train(graph, metadata)
            train_loss.append(loss.item())
        
        val_loss, val_auc = trainer.eval()

        train_loss = torch.tensor(train_loss).mean()

        logger.info(
            f"epoch: {epoch}, loss: {train_loss:10.3e}, val_loss: {val_loss:10.3e}, val_auc = {val_auc:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )

        # save checkpoint
        save_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=epoch,
        )
        start = time.time()
        trainer.scheduler.step()

        with open(cfg.checkpoints.ckpt_path + "/parameters.json", "w") as outf:
            json.dump(trainer.params, outf, indent=4)
    logger.info("Training completed!")


"""
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """
if __name__ == "__main__":
    do_training()