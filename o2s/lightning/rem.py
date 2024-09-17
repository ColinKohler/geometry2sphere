from typing import Optional, Dict, List

import pytorch_lightning as pl
import torch as tr
from torch import Tensor, nn
from torchmetrics import MeanSquaredError, MetricCollection

from o2s.metrics.metrics import (
    calculated_base_matching_score,
    maxima_matching_score,
    calculated_peak_matching_score,
)

from o2s.typing import PartialOptimDict
from o2s.lightning._base import _BaseModule
import logging

log = logging.getLogger(__name__)


class REMLightningModule(_BaseModule, pl.LightningModule):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        criterion: nn.Module,
        optim: PartialOptimDict,
        log_results_every_n_epochs: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.optim = optim
        self.criterion = criterion
        self.metrics_epoch = {}
        self.log_results_every_n_epochs = log_results_every_n_epochs

        self.metrics = MetricCollection(
            dict(
                mse=MeanSquaredError(),
            )
        )

    def calculate_mse(self, pred, target):
        mse = (pred - target) ** 2
        return mse.mean()

    def calculate_losses(
        self,
        batch: Dict[str, Tensor],
        stage: str,
        advanced_metrics: bool = False,
        ks: List[int] = [1, 2],
    ):
        loss = self.calculate_equivariant_loss(batch, stage, advanced_metrics, ks)

        return loss

    def calculate_equivariant_loss(self, batch, stage, advanced_metrics, ks):
        data, target = batch
        B, R, T, P = target.shape

        pred, _ = self.forward(data)

        loss = 0
        mse = 0
        loss += self.criterion(pred, target)
        mse += self.calculate_mse(pred, target)

        if stage == "train":
            lr = 0
            for group in self.optimizers().optimizer.param_groups:
                lr = group["lr"]
            self.log("lr", lr, prog_bar=True, on_step=True)

        self.log(
            f"{stage}/loss",
            loss,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=B,
        )
        self.log(
            f"{stage}/mse_error",
            mse,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=B,
            prog_bar=True,
        )

        if advanced_metrics:
            base_score = 0
            maxima_val_score, maxima_dist_score = [0 for k in ks], [0 for k in ks]
            peak_val_score, peak_dist_score = [0 for k in ks], [0 for k in ks]
            base_score += calculated_base_matching_score(target, pred)

            for i, k in enumerate(ks):
                mvs, mds = maxima_matching_score(target, pred, k=k)
                pvs, pds = calculated_peak_matching_score(target, pred, max_num_peaks=k)
                maxima_val_score[i] += mvs
                maxima_dist_score[i] += mds
                peak_val_score[i] += pvs
                peak_dist_score[i] += pds

            self.log(
                f"{stage}/base_score",
                base_score,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
                batch_size=B,
            )
            for i, k in enumerate(ks):
                self.log(
                    f"{stage}/maxima_val_score_k{k}",
                    maxima_val_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=B,
                )
                self.log(
                    f"{stage}/maxima_dist_score_k{k}",
                    maxima_dist_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=B,
                )
                self.log(
                    f"{stage}/peak_val_score_k{k}",
                    peak_val_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=B,
                )
                self.log(
                    f"{stage}/peak_dist_score_k{k}",
                    peak_dist_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=B,
                )

        return loss


class SoftmaxWeightedMSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        B, R, T, P = target.shape
        weight = nn.functional.softmax(target.reshape(B * R, -1), dim=1).reshape(
            B, R, T, P
        )
        loss = weight * (pred - target) ** 2

        weighted_mse_loss = (loss / (B * R * T * P)).sum()
        # weighted_mse_loss = (weight * (pred - target) ** 2).mean()
        mse_loss = nn.functional.mse_loss(pred, target)

        loss = weighted_mse_loss + mse_loss
        return loss
