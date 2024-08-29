from typing import Optional, Callable

import pytorch_lightning as pl
from torch_geometric.data import DataLoader, Dataset
from torch.utils.data import DataLoader as StandardDataLoader
import logging

log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
        dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
        geometric_dataloader: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.geometric_dataloader = geometric_dataloader
        if self.geometric_dataloader:
            self.dataloader_class = DataLoader
        else:
            self.dataloader_class = StandardDataLoader

    def train_dataloader(self) -> Optional[DataLoader]:
        log.info('returning training dataloader')
        if self.dataset is not None:
            return self.dataloader_class(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )

    def val_dataloader(self) -> Optional[DataLoader]:
        log.info('returning val dataloader')
        if self.val_dataset is not None:
            return self.dataloader_class(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self) -> Optional[DataLoader]:
        log.info('returning test dataloader')
        if self.test_dataset is not None:
            return self.dataloader_class(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self) -> Optional[DataLoader]:
        log.info('returning predict dataloader')
        if self.test_dataset is not None:
            return self.dataloader_class(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
