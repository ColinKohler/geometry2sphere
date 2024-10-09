from typing import Dict, Optional, List

import numpy as np
import json
import pytorch_lightning as pl
import torch as tr
from torch import Tensor, nn
from pathlib import Path
from abc import ABC, abstractmethod

from ..metrics.metrics import calculated_base_matching_score, maxima_matching_score, calculated_peak_matching_score
from ..typing import PartialOptimDict
from ._base import _BaseLogging, _BaseModule
import logging
from pytorch_lightning.loggers import MLFlowLogger

log = logging.getLogger(__name__)

class SoftmaxWeightedMSELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature=temperature

    def forward(self, pred, target):
        weight = nn.functional.softmax(target/self.temperature, dim = -1)
        loss = weight*(pred - target)**2
        batch_size, num_orientations, range_bins = pred.shape

        weighted_mse_loss = (loss/(batch_size*num_orientations)).sum()
        mse_loss = nn.functional.mse_loss(pred, target)

        loss = weighted_mse_loss + mse_loss

        return loss


class BaselineTransformerLightningModule(_BaseModule, _BaseLogging, pl.LightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        criterion: nn.Module,
        optim: PartialOptimDict,
        logger: Optional[MLFlowLogger],
        log_results_every_n_epochs: Optional[int] = 20,
        **kwargs,
    ) -> None:
        # super().__init__(
        #     backbone=backbone, 
        #     criterion=criterion, 
        #     optim=optim, 
        #     # log_results_every_n_epochs=log_results_every_n_epochs,
        #     logger=logger,
        #     )

        super().__init__()

        self.backbone = backbone
        self.criterion = criterion
        self.optim = optim
        self.criterion = criterion
        self.metrics_epoch = {}
        self.log_results_every_n_epochs = log_results_every_n_epochs
        self.output_dir = None

        self.results = []

    def set_output_dir(self, dir):
        log.info('setting output dir')
        self.output_dir = dir

    def calculate_mse_error(self, pred, target):
        mse = ((pred - target)**2)
        return mse.mean()

    #overrides logging behavior
    def on_fit_end(self):
        pass

class BaselineTransformerRadarLightningModule(BaselineTransformerLightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        criterion: nn.Module,
        optim: PartialOptimDict,
        logger: Optional[MLFlowLogger],
        log_results_every_n_epochs: Optional[int] = 20,
    ) -> None:
        super().__init__(
            backbone=backbone, 
            criterion=criterion, 
            optim=optim, 
            log_results_every_n_epochs=log_results_every_n_epochs,
            logger=logger
            )
    
    def calculate_losses(self, batch: Dict[str, Tensor], stage: str, advanced_metrics:bool = True, ks:List[int]=[1,2]):
        target_profile = batch['data']
        pred_range_profile = self.forward(batch)
        print(pred_range_profile.shape)
        loss = self.criterion(pred_range_profile, target_profile)
        if self.loss_scaling is not None:
            loss = loss*self.loss_scaling

        mse_error = self.calculate_mse_error(pred_range_profile, target_profile)

        self.log(f"{stage}/loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/mse_error", mse_error, sync_dist=True, on_epoch=True, on_step=False)

        if advanced_metrics:
            base_score = calculated_base_matching_score(target_profile, pred_range_profile)
            self.log(f"{stage}/base_score", base_score, sync_dist=True, on_epoch=True, on_step=False)

            for k in ks:
                maxima_val_score, maxima_dist_score = maxima_matching_score(target_profile, pred_range_profile, k=k)
                self.log(f"{stage}/maxima_val_score_k{k}", maxima_val_score, sync_dist=True, on_epoch=True, on_step=False)
                self.log(f"{stage}/maxima_dist_score_k{k}", maxima_dist_score, sync_dist=True, on_epoch=True, on_step=False)


                peak_val_score, peak_dist_score = calculated_peak_matching_score(target_profile, pred_range_profile, max_num_peaks=k)
                self.log(f"{stage}/peak_val_score_k{k}", peak_val_score, sync_dist=True, on_epoch=True, on_step=False)
                self.log(f"{stage}/peak_dist_score_k{k}", peak_dist_score, sync_dist=True, on_epoch=True, on_step=False)


        return loss
    
    def test_step(self, batch, _):
        with tr.no_grad():
            pred_range_profile = self.forward(batch)

        return dict(
            predic=pred_range_profile.detach().cpu().numpy(),
            target=batch['data'].detach().cpu().numpy(),
            seed=batch['seed'].detach().cpu().numpy(),
        )

    def predict_step(self, batch, _) -> Dict[str, Tensor]:
        with tr.no_grad():
            pred_range_profile = self.forward(batch)

        batch_results =  dict(
            predic=pred_range_profile.detach().cpu().numpy(),
            target=batch['data'].detach().cpu().numpy(),
            seed=batch['seed'].detach().cpu().numpy(),
            label=batch['label'].detach().cpu().numpy(),
        )

        self.results.append(batch_results)

        return batch_results
