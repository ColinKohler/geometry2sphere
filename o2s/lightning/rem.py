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
from _base import _BaseLogging, _BaseModule
import logging

log = logging.getLogger(__name__)


class REMLightningModule(_BaseModule, _BaseLogging, pl.LightningModule):
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
        advanced_metrics: bool = True,
        ks: List[int] = [1, 2],
    ):
        if self.backbone.invariant_out:
            loss = self.calculate_invariant_loss(
                batch, stage, advanced_metrics, ks)
        else:
            loss = self.calculate_equivariant_loss(
                batch, stage, advanced_metrics, ks)

        return loss

    def calculate_equivariant_loss(self, batch, stage, advanced_metrics, ks):
        data, poses, target = batch
        b = poses.size(0)
        pred = self.forward(data)

        loss = 0
        mse = 0
        pred_range_profile = self.backbone.getResponse(
            pred, poses.to(tr.float32))
        loss += self.criterion(pred_range_profile, target)
        mse += self.calculate_mse(pred_range_profile, target)

        if stage == 'train':
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
            batch_size=b,
        )
        self.log(
            f"{stage}/mse_error",
            mse,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=b,
            prog_bar=True,
        )

        if advanced_metrics:
            base_score = 0
            maxima_val_score, maxima_dist_score = [
                0 for k in ks], [0 for k in ks]
            peak_val_score, peak_dist_score = [0 for k in ks], [0 for k in ks]
            base_score += calculated_base_matching_score(
                target, pred_range_profile)

            for i, k in enumerate(ks):
                mvs, mds = maxima_matching_score(
                    target, pred_range_profile, k=k
                )
                pvs, pds = calculated_peak_matching_score(
                    target, pred_range_profile, max_num_peaks=k
                )
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
                batch_size=b,
            )
            for i, k in enumerate(ks):
                self.log(
                    f"{stage}/maxima_val_score_k{k}",
                    maxima_val_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=b,
                )
                self.log(
                    f"{stage}/maxima_dist_score_k{k}",
                    maxima_dist_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=b,
                )
                self.log(
                    f"{stage}/peak_val_score_k{k}",
                    peak_val_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=b,
                )
                self.log(
                    f"{stage}/peak_dist_score_k{k}",
                    peak_dist_score[i],
                    sync_dist=True,
                    on_epoch=True,
                    on_step=False,
                    batch_size=b,
                )

        return loss


class SoftmaxWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        weight = nn.functional.softmax(target, dim=1)
        loss = weight * (pred - target) ** 2
        return loss.sum()


def get_data_transitions(data):
    max_vals, _ = data.abs().max(dim=1)
    data = data / (
        max_vals[:, None] + 1e-7
    )  # normalize by max value to ensure measuring relative smoothness, not size
    transitions = (data[:, :-1] - data[:, 1:]).abs()
    return transitions


class SoftmaxWeightedSmoothingMSELoss(nn.Module):
    def __init__(self, mse_weight=1):
        super().__init__()

        self.mse_weight = mse_weight

    def forward(self, pred, target):
        # loss = (pred - target)**2 * (target*30+0.2)
        # calculate MSE loss
        weight = nn.functional.softmax(target, dim=1)
        mse_loss = (weight * (pred - target) ** 2).sum()

        # calculate smoothing loss
        transitions_per_sample = pred.numel() - pred.size()[0]

        pred_transitions = get_data_transitions(pred)
        target_transitions = get_data_transitions(target)

        transition_mse_loss = (
            (pred_transitions - target_transitions) ** 2
        ).sum() / transitions_per_sample

        loss = self.mse_weight * mse_loss + transition_mse_loss

        return loss