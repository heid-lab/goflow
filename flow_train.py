from typing import List
import torch
import lightning as L
from lightning import LightningDataModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
import hydra
from omegaconf import DictConfig
from flow_matching.flow_module import FlowModule

from gotennet.utils import RankedLogger, instantiate_callbacks, instantiate_loggers, log_hyperparameters

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def train_flow(cfg: DictConfig):
    #################### PyTorch Specifics ####################
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    if cfg.get("seed"): L.seed_everything(cfg.seed, workers=True)

    #################### Load Modules for Data/Model/Callback/Logging ####################
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: FlowModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    #################### Run Training or Testing ####################
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = cfg.get("custom_model_weight_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


if __name__ == '__main__':
    train_flow()
