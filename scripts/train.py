from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch_geometric.data import Dataset
from omegaconf import DictConfig
from hydra.utils import instantiate
from o2s.utils.logging_utils import log_hyperparameters
from o2s.lightning.data import DataModule
from rai_toolbox.mushin.lightning.launchers import HydraConfig


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    pl.seed_everything(42)
    train_dataset: Dataset = instantiate(cfg.train_dataset)
    val_dataset: Dataset = instantiate(cfg.val_dataset)
    test_dataset: Dataset = instantiate(cfg.test_dataset)
    datamodule: LightningDataModule = DataModule(
        dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    module: LightningModule = hydra.utils.instantiate(cfg.module)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "model": module,
        "trainer": trainer,
    }

    log_hyperparameters(object_dict)

    trainer.fit(model=module, datamodule=datamodule)
    trainer.test(model=module, datamodule=datamodule, ckpt_path="best")
    module.save_metrics()


@hydra.main(version_base="1.3", config_path="../config", config_name="rem_config.yaml")
def main(cfg: DictConfig) -> None:

    train(cfg)


if __name__ == "__main__":
    main()
