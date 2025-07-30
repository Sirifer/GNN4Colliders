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
from dataset import get_dataset
import json

import random

from sklearn.metrics import roc_auc_score

import models.MeshGraphNet as MeshGraphNet

import torch.nn.functional as F

def bce(input, target, device=None, weights=None):
    if input.shape != target.shape:
        if input.shape[-1] == 1 and input.shape[:-1] == target.shape:
            input = input.squeeze(-1)
        elif target.shape[-1] == 1 and target.shape[:-1] == input.shape:
            target = target.squeeze(-1)

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return torch.mean(loss)

def weighted_bce(input, target, device=None, weights=None):
    """
    Compute a weighted and label-normalized binary cross entropy (BCE) loss.

    For each unique label in the target tensor, the BCE loss is computed and weighted,
    then normalized by the sum of weights for that label. The final loss is the mean
    of these per-label normalized losses.

    Args:
        input (Tensor): Predicted logits of shape (N, ...).
        target (Tensor): Ground truth labels of shape (N, ...), with discrete label values.
        device (torch.device or None): Device to move tensors to (optional).
        weights (Tensor or None): Optional tensor of per-sample weights, same shape as input/target.

    Returns:
        Tensor: Scalar tensor representing the normalized weighted BCE loss.
    """

    if input.shape != target.shape:
        if input.shape[-1] == 1 and input.shape[:-1] == target.shape:
            input = input.squeeze(-1)
        elif target.shape[-1] == 1 and target.shape[:-1] == input.shape:
            target = target.squeeze(-1)

    if weights is None:
        weights = torch.ones_like(target).to(device)

    # Compute per-element BCE loss (no reduction)
    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

    # Vectorized label normalization
    unique_labels = torch.unique(target)
    normalized_losses = []
    for label in unique_labels:
        label_mask = (target == label)
        label_weights = weights[label_mask]
        label_losses = loss[label_mask]
        weight_sum = label_weights.sum()
        if weight_sum > 0:
            label_loss = (label_weights * label_losses).sum() / weight_sum
            normalized_losses.append(label_loss)

    if normalized_losses:
        return torch.stack(normalized_losses).mean()
    else:
        return torch.tensor(0.0, device=input.device)

class MGNTrainer:
    def __init__(self, logger, cfg, dist):
        # set device
        self.device = dist.device
        logger.info(f"Using {self.device} device")

        params = {}

        start = time.time()
        self.dataloader, self.valloader, self.testloader = get_dataset(cfg, self.device)
        print(f"total time loading dataset: {time.time() - start:.2f} seconds")

        dtype_str = getattr(cfg.root_dataset, "type", "torch.float32")
        if isinstance(dtype_str, str) and dtype_str.startswith("torch."):
            self.dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
        else:
            self.dtype = torch.float32

        nodes_features = cfg.root_dataset.features
        edges_features = ["dR", "deta", "dphi"]
        global_features = ["num_nodes"]

        params["infeat_nodes"] = len(nodes_features)
        params["infeat_edges"] = len(edges_features)
        params["infeat_globals"] = len(global_features)
        params["out_dim"] = cfg.architecture.hidden_dim_node_encoder
        params["node_features"] = list(nodes_features)
        params["edges_features"] = edges_features
        params["global_features"] = global_features

        self.model = MeshGraphNet.MeshGraphNet(
            params["infeat_nodes"],
            params["infeat_edges"],
            params['out_dim'],
            processor_size=cfg.architecture.processor_size,
            hidden_dim_node_encoder=cfg.architecture.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.architecture.hidden_dim_edge_encoder,
            hidden_dim_processor=cfg.architecture.hidden_dim_processor,
            hidden_dim_node_decoder=cfg.architecture.hidden_dim_node_decoder,
        )
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

    def train(self, graph, label):
        """
        Perform one training iteration over one graph. The training is performed
        over multiple timesteps, where the number of timesteps is specified in
        the 'stride' parameter.

        Arguments:
            graph: the desired graph.

        Returns:
            loss: loss value.

        """
        self.optimizer.zero_grad()
        pred = self.model(graph.ndata["features"], graph.edata["features"], graph)
        loss = bce(pred, label, device=self.device)
        self.backward(loss)
        return loss

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

        for graph, label in self.valloader:

            graph = graph.to(self.device)
            pred = self.model(graph.ndata["features"], graph.edata["features"], graph)
            predictions.append(pred)
            labels.append(label)     

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

        loss = weighted_bce(predictions, labels, device=self.device)
        
        # Convert logits to probabilities
        prob = torch.sigmoid(predictions)

        # Flatten to 1D arrays
        prob_flat = prob.detach().to(torch.float32).cpu().numpy().flatten()
        labels_flat = labels.detach().to(torch.float32).cpu().numpy().flatten()

        # Calculate AUC
        try:
            auc = roc_auc_score(labels_flat, prob_flat)
        except ValueError:
            auc = float('nan')  # Not enough classes present for AUC

        return loss, auc

@hydra.main(version_base=None, config_path=".", config_name="config")
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
    logger = PythonLogger("main")
    logger.file_logging()

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
        for graph, label in trainer.dataloader:
            trainer.model.train()
            train_loss.append(trainer.train(graph, label))
        
        val_loss, val_auc = trainer.eval()

        train_loss = torch.mean(torch.stack(train_loss)).item()

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