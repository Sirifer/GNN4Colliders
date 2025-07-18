import time, os

start = time.time()
import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import hydra
from omegaconf import DictConfig
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.distributed.manager import DistributedManager
from Dataset import get_dataset
import json

class MGNTrainer:
    def __init__(self, logger, cfg, dist):
        # set device
        self.device = dist.device
        logger.info(f"Using {self.device} device")

        norm_type = {"features": "normal", "labels": "normal"}

        self.train_loader, self.val_loader, self.test_loader = get_dataset(cfg)
        print(f"train: {self.train_loader}")
        print(f"val: {self.val_loader}")
        print(f"test: {self.test_loader}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def do_training(cfg: DictConfig):
    """
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    logger = PythonLogger("main")
    logger.file_logging()

    # initialize trainer
    trainer = MGNTrainer(logger, cfg, dist)

    # training loop
    start = time.time()
    logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        for graph in trainer.dataloader:
            loss = trainer.train(graph)

        logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
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