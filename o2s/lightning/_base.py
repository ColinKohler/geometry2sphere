from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from abc import ABC, abstractmethod

from torch import Tensor
import json
import pytorch_lightning as pl
import hydra
from hydra_zen import load_from_yaml
from omegaconf import ListConfig
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn

from o2s.lightning.lr_scheduler import get_cosine_schedule_with_warmup, cosinewithWarmUp
from o2s.typing import OptimDict, PartialOptimDict
import logging

log = logging.getLogger(__name__)

class _BaseModule(ABC):
    backbone: nn.Module
    optim: Optional[PartialOptimDict]
    criterion: nn.Module
    logger: Optional[MLFlowLogger]

    def forward(self, batch, **kwargs):
        return self.backbone(batch, **kwargs)

    @abstractmethod
    def calculate_losses(
        self,
        batch: Dict[str, Tensor],
        stage: str,
        **kwargs
    ):
        pass

    def training_step(self, batch, _):
        return self.calculate_losses(batch, stage="train")

    def validation_step(self, batch, _):
        return self.calculate_losses(batch, stage="val")

    def test_step(self, batch, _):
        return self.calculate_losses(batch, stage="test")

    def predict_step(self, batch, _) -> Dict[str, Tensor]:
        pred_range_profile = self.forward(batch[0])

        return dict(
            pred_range_profile=pred_range_profile.detach().cpu().numpy(),
            poses=batch[1],
            target_range_profile=batch[2],
            label=batch[3],
            segments=batch[4]
        )

    def save_metrics(self):
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        metric = Path(hydra_output_dir + "metric.json")
        with open(metric, "w") as f:
            json.dump(self.metrics_epoch, f)

    def configure_optimizers(self) -> Optional[OptimDict]:
        if self.optim is not None:
            frequency = self.optim.get("frequency", 1)

            assert "optimizer" in self.optim and self.optim["optimizer"] is not None
            optimizer = self.optim["optimizer"](self.backbone.parameters())

            lr_scheduler = None
            if "lr_scheduler" in self.optim:
                if isinstance(self.optim["lr_scheduler"], cosinewithWarmUp):
                    num_train_steps, num_warmup_steps = self.compute_warmup(
                        num_training_steps=-1, num_warmup_steps=self.optim["lr_scheduler"].num_warmup_steps
                    )
                    print(f"No. warmup steps: {num_warmup_steps}")
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
                    )
                elif self.optim["lr_scheduler"] is not None:
                    lr_scheduler = self.optim["lr_scheduler"](optimizer)

            opt: OptimDict = OptimDict(
                frequency=frequency, optimizer=optimizer, lr_scheduler=lr_scheduler
            )

            return opt

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches
