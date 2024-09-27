from typing import Any, Dict, List, Optional, Tuple

import os
import hydra
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch_geometric.data import Dataset
from omegaconf import DictConfig
from hydra.utils import instantiate
from o2s.lightning.data import DataModule
from o2s.datasets.transformer_drag_dataset import transformer_collate_function


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
        collate_fn=transformer_collate_function,
        geometric_dataloader=False,
    )
    module: LightningModule = hydra.utils.instantiate(cfg.module)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Hydra output dir: {output_dir}")
    trainer.logger.experiment.log_param(trainer.logger.run_id, "output_dir", output_dir)

    trainer.checkpoint_callback.dirpath = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )  # Saves model in same dir as hydra output dir
    trainer.fit(model=module, datamodule=datamodule)
    trainer.test(model=module, datamodule=datamodule, ckpt_path="best")


@hydra.main(
    version_base="1.3", config_path="../config", config_name="rem_basic_asym.yaml"
)
def main(cfg: DictConfig) -> None:

    train(cfg)


if __name__ == "__main__":
    main()
