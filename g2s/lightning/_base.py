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

from g2s.lightning.lr_scheduler import get_cosine_schedule_with_warmup, cosinewithWarmUp
from g2s.typing import OptimDict, PartialOptimDict
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
    def calculate_losses(self, batch: Dict[str, Tensor], stage: str, **kwargs):
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
            segments=batch[4],
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
                        num_training_steps=-1,
                        num_warmup_steps=self.optim["lr_scheduler"].num_warmup_steps,
                    )
                    print(f"No. warmup steps: {num_warmup_steps}")
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_train_steps,
                    )
                elif self.optim["lr_scheduler"] is not None:
                    lr_scheduler = self.optim["lr_scheduler"](optimizer)

            opt: OptimDict = OptimDict(
                frequency=frequency, optimizer=optimizer, lr_scheduler=lr_scheduler
            )

            return opt

    def compute_warmup(
        self, num_training_steps: int, num_warmup_steps: Union[int, float]
    ) -> Tuple[int, int]:
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


class _BaseLogging:
    logger: Optional[MLFlowLogger]
    trainer: pl.Trainer

    def _mlflow_load_params(self):
        assert isinstance(self.logger, MLFlowLogger)

        cfg = Path.cwd() / ".hydra/config.yaml"
        self.logger.experiment.log_artifact(
            self.logger.run_id, str(cfg), artifact_path="configs"
        )

        hydra_cfg = Path.cwd() / ".hydra/hydra.yaml"
        self.logger.experiment.log_artifact(
            self.logger.run_id, str(hydra_cfg), artifact_path="configs"
        )

        overrides_cfg = Path.cwd() / ".hydra/overrides.yaml"
        self.logger.experiment.log_artifact(
            self.logger.run_id, str(overrides_cfg), artifact_path="configs"
        )

        choices_logged = []
        hydra_cfg = load_from_yaml(hydra_cfg)
        for choice, value in hydra_cfg.hydra.runtime.choices.items():
            if "hydra/" in choice:
                continue

            choices_logged.append(choice)

            self.logger.experiment.log_param(
                self.logger.run_id,
                choice.replace("/", "_"),
                value,
            )

        overrides = load_from_yaml(overrides_cfg)
        assert isinstance(overrides, ListConfig)
        for o in overrides:
            param, val = o.split("=", 1)
            param = param.replace("+", "")

            if param in choices_logged:
                continue

            param = param.replace("/", "_")
            self.logger.experiment.log_param(self.logger.run_id, param, val)

    def on_fit_start(self) -> None:
        self._mlflow_load_params()

    def on_test_start(self) -> None:
        self._mlflow_load_params()

    def _log_checkpoint(self):
        log.info("Logging checkpoint function called")
        log.info(self.trainer.current_epoch)
        if self.trainer.current_epoch % 10 == 0:
            assert isinstance(self.logger, MLFlowLogger)
            from pathlib import Path

            ckpts = list(Path.cwd().glob("**/*.ckpt"))
            path = str(Path.cwd())
            log.info(f"ckpts found: {len(ckpts)}")
            log.info(f"ckpt path: {path}")
            if len(ckpts) >= 1:
                ckpt = ckpts[0]
                log.info(f"saving {str(ckpt)}")
                self.logger.experiment.log_artifact(
                    self.logger.run_id, str(ckpt), artifact_path="models"
                )

    def on_fit_end(self):
        pass
        # self._log_checkpoint()
