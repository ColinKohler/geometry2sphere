""" workflows.py """

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional
import os
import pickle
import csv
from tqdm import tqdm

import pytorch_lightning as pl
import torch as tr
import numpy as np
from hydra_zen import instantiate, launch, load_from_yaml
from mlflow import MlflowClient, set_tracking_uri
import logging

from baselines import user_configs
from baselines.metrics.metrics import (
    maxima_matching_score,
    calculated_peak_matching_score,
    calculated_base_matching_score,
    convert_log_center_to_original,
    convert_log_center_to_db,
)
from baselines.lightning.data import ExperimentDataModule

log = logging.getLogger(__name__)


class _BaseWorkflow:
    @staticmethod
    def pre_task(global_seed: int, mlflow_tracking_uri: Optional[str] = None):
        pl.seed_everything(global_seed)

        if mlflow_tracking_uri is not None:
            set_tracking_uri(mlflow_tracking_uri)

    @staticmethod
    def task(cfg):
        pass

    @classmethod
    def run(
        cls,
        cfg,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        to_dictconfig: bool = True,
        version_base: str = "1.1",
    ):
        launch_overrides = []
        if overrides is not None:
            for k, v in overrides.items():
                value_check(
                    k, v, type_=(int, float, bool, str, dict, multirun, hydra_list)
                )
                if isinstance(v, multirun):
                    v = ",".join(str(item) for item in v)

                launch_overrides.append(f"{k}={v}")
        return launch(
            cfg,
            _task_calls(
                pre_task=zen(cls.pre_task),
                task=zen(cls.task),
            ),
            overrides=launch_overrides,
            multirun=True,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
        )


class TrainingWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        run_id=None,
    ):

        # config logging taken care of by base logging class
        # log path to model to mlflow
        hydra = load_from_yaml(".hydra/hydra.yaml")
        output_dir = hydra.hydra.runtime.output_dir
        trainer.logger.experiment.log_param(
            trainer.logger.run_id, "output_dir", output_dir
        )

        # Load checkpoint if run_id was passed
        if run_id:
            mc = MlflowClient(user_configs.RESULTS_PATH)
            run = mc.get_run(run_id)
            prev_output_dir = run.data.params["output_dir"]
            ckpt_path = Path(prev_output_dir) / "last.ckpt"
        else:
            ckpt_path = None

        tr.backends.cudnn.benchmark = True
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(model=module, datamodule=datamodule, ckpt_path="best")
        module.save_metrics()


class TestingWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ):
        ckpt = glob.glob(
            "/home/gridsan/RA31303/repos/baselines/experiments/mesh_k5_4layers/0/*.ckpt"
        )[0]
        trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt)
        module.save_metrics()


class PredictionWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        global_seed: int,
        mlflow_tracking_uri: str,
        run_id: str,
        trainer: pl.Trainer,
        datamodule: pl.LightningDataModule,
        mlflow_experiment_name: str,
        mlflow_run_name: str,
    ):

        # load training config and model checkpoint
        mc = MlflowClient(mlflow_tracking_uri)
        run = mc.get_run(run_id)
        info = run.info

        cfg_file = mc.download_artifacts(info.run_id, "configs/config.yaml")
        ckpt_file = mc.download_artifacts(info.run_id, "models/model-v1.ckpt")

        # instantiate model and load weights
        cfg = load_from_yaml(cfg_file)
        module = instantiate(cfg.module).load_from_checkpoint(
            ckpt_file,
            backbone=instantiate(cfg.module.backbone),
            criterion=instantiate(cfg.module.criterion),
        )

        # run predictions
        outputs = trainer.predict(module, datamodule=datamodule)
        assert outputs is not None
        assert len(outputs) >= 1
        assert isinstance(outputs[0], dict)

        # Log results in MLFlow under the "predictions" experiment
        exp = mc.get_experiment_by_name(mlflow_experiment_name)
        if exp is None:
            exp_id = mc.create_experiment(name=mlflow_experiment_name)
        else:
            exp_id = exp.experiment_id
        prun = mc.create_run(exp_id, tags={"mlflow.runName": mlflow_run_name})

        mc.log_param(prun.info.run_id, "training_run_id", run_id)
        mc.log_param(prun.info.run_id, "test_global_seed", global_seed)

        for k, v in run.data.params.items():
            mc.log_param(prun.info.run_id, f"training_{k}", v)

        # create Xarray Dataset
        from mlflow.entities import Metric

        preds = defaultdict(list)
        for o in outputs:
            for k, v in o.items():
                preds[k].append(v)

        for k, v in preds.items():
            v = tr.cat(v, 0)
            if v.ndim == 1:
                metrics = []
                for i in range(len(v)):
                    vi = v[i].item()
                    if isinstance(vi, bool):
                        vi = int(vi)
                    metrics.append(Metric(k, vi, timestamp=i, step=i))

                mc.log_batch(prun.info.run_id, metrics)


class PostProcessingPredictionWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        global_seed: int,
        mlflow_tracking_uri: str,
        run_id: str,
        trainer: pl.Trainer,
        datamodule: pl.LightningDataModule,
        mlflow_experiment_num: int,
        untransform_pred: bool,
        metrics_prefix: str = None,
    ):

        # load training config and model checkpoint
        log.info("current directory")
        log.info(os.getcwd())

        set_tracking_uri(mlflow_tracking_uri)
        mc = MlflowClient()
        all_runs = MlflowClient().search_runs(
            experiment_ids=str(mlflow_experiment_num),
        )

        run = [run for run in all_runs if run.info.run_id == run_id][0]
        info = run.info

        output_dir = run.data.params["output_dir"]
        cfg_file = str(list(Path(output_dir).glob("**/config.yaml"))[0])
        ckpt_file = [
            c for c in list(Path(output_dir).glob("**/*.ckpt")) if "last" not in str(c)
        ][
            0
        ]  # selecting best val error, not last model

        # instantiate model and load weights
        cfg = load_from_yaml(cfg_file)
        module = instantiate(cfg.module).load_from_checkpoint(
            ckpt_file,
            backbone=instantiate(cfg.module.backbone),
            criterion=instantiate(cfg.module.criterion),
            optim=instantiate(cfg.module.optim),
        )
        module.eval()

        module.set_output_dir(os.getcwd())
        log.info("module loaded and output_dir set")

        test_dl = datamodule.predict_dataloader()

        log.info("dataset length")
        log.info(len(test_dl.dataset))
        # run predictions
        _ = trainer.predict(module, datamodule=datamodule)
        log.info("prediction complete")

        pred_files = [d for d in os.listdir(os.getcwd()) if ".pt" in d]
        outputs = []
        for f in pred_files:
            individual_outputs = tr.load(f)
            for o in individual_outputs:
                outputs.append(o)

        assert outputs is not None
        assert len(outputs) >= 1
        assert isinstance(outputs[0], dict)

        targets = [tr.tensor(output["target"]) for output in outputs]
        predics = [tr.tensor(output["predic"]) for output in outputs]

        full_targets = tr.cat(targets).squeeze()
        full_predictions = tr.cat(predics).squeeze()

        log.info("prediction sizes")
        log.info(full_predictions.size())

        if untransform_pred:  # TODO: handle this without assume training on log center
            log.info("transforming pred and data to original")
            full_targets = convert_log_center_to_original(
                full_targets, std=cfg.root_config.std, mean=cfg.root_config.mean
            ).detach()
            full_predictions = convert_log_center_to_original(
                full_predictions, std=cfg.root_config.std, mean=cfg.root_config.mean
            ).detach()

        with tr.no_grad():
            size = full_targets.size()[0]
            batch_size = 1000
            idx_window_starts = np.arange(0, size, batch_size).tolist()
            idx_window_ends = idx_window_starts[1:] + [-1]
            idx_windows = [(s, e) for s, e in zip(idx_window_starts, idx_window_ends)]

            full_metrics = defaultdict(list)

            for start_idx, end_idx in idx_windows:
                if end_idx != -1:
                    targets = full_targets[start_idx:end_idx]
                    preds = full_predictions[start_idx:end_idx]
                else:
                    targets = full_targets[start_idx:]
                    preds = full_predictions[start_idx:]

                base_score = calculated_base_matching_score(targets, preds)
                maxima_val_score1, maxima_dist_score1 = maxima_matching_score(
                    targets, preds, k=1
                )
                peak_val_score1, peak_dist_score1 = calculated_peak_matching_score(
                    targets, preds, max_num_peaks=1
                )
                maxima_val_score2, maxima_dist_score2 = maxima_matching_score(
                    targets, preds, k=2
                )
                peak_val_score2, peak_dist_score2 = calculated_peak_matching_score(
                    targets, preds, max_num_peaks=2
                )
                mse_error = tr.nn.functional.mse_loss(
                    targets.squeeze(), preds.squeeze()
                )

                metrics = {
                    f"test/mse_error": mse_error,
                    f"test/base_score": base_score,
                    f"test/maxima_val_score_1": maxima_val_score1,
                    f"test/maxima_dist_score_1": maxima_dist_score1,
                    f"test/peak_val_score_1": peak_val_score1,
                    f"test/peak_dist_score_1": peak_dist_score1,
                    f"test/maxima_val_score_2": maxima_val_score2,
                    f"test/maxima_dist_score_2": maxima_dist_score2,
                    f"test/peak_val_score_2": peak_val_score2,
                    f"test/peak_dist_score_2": peak_dist_score2,
                }

                for k, v in metrics.items():
                    full_metrics[k].append(v)

            if metrics_prefix is not None:
                metrics = {
                    f"{metrics_prefix}/{k}": tr.stack(v).mean().item()
                    for k, v in full_metrics.items()
                }
            else:
                metrics = {
                    k: tr.stack(v).mean().item() for k, v in full_metrics.items()
                }

        log.info("logging metrics")
        for k, v in metrics.items():
            mc.log_metric(info.run_id, k, v)
            msg = f"{k}: {v}"
            log.info(msg)

        log.info("saving_predictions")
        save_dir = Path(output_dir) / "saved_predictions"
        log.info(str(save_dir))

        os.makedirs(save_dir, exist_ok=True)
        tr.save(full_targets, str(save_dir / "targets.pt"))
        tr.save(full_predictions, str(save_dir / "predictions.pt"))


def saveCsvMetrics(sample_error):
    with open("sample_metrics.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(
            [
                "mesh",
                "los",
                "mse",
                "base",
                "maxima_val",
                "maxima_bin",
                "peak_val",
                "peak_bin",
            ]
        )
        for fk in sample_error.keys():
            for lk in sample_error[fk].keys():
                csv_writer.writerow(
                    [
                        fk,
                        lk,
                        sample_error[fk][lk]["mse"].item(),
                        sample_error[fk][lk]["base_val"].item(),
                        sample_error[fk][lk]["maxima_val"].item(),
                        sample_error[fk][lk]["maxima_bin"].item(),
                        sample_error[fk][lk]["peak_val"].item(),
                        sample_error[fk][lk]["peak_bin"].item(),
                    ]
                )


def convertLogCenterToDb(data, std, mean):
    return ((data * std) + mean) * 10


class ColinsPredictionWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        mlflow_experiment_num,
        run_id,
        mlflow_tracking_uri,
        datamodule,
        save_metrics_per_sample=False,
        runs=None,
        batch_size=64,
        untransform_pred=False,
    ):

        set_tracking_uri(mlflow_tracking_uri)
        all_runs = MlflowClient().search_runs(
            experiment_ids=str(mlflow_experiment_num),
        )

        run = [run for run in all_runs if run.info.run_id == run_id][0]

        output_dir = run.data.params["output_dir"]
        cfg_file = str(list(Path(output_dir).glob("**/config.yaml"))[0])
        ckpt_file = [
            c for c in list(Path(output_dir).glob("**/*.ckpt")) if "last" not in str(c)
        ][
            0
        ]  # selecting best val error, not last model

        # instantiate model and load weights
        cfg = load_from_yaml(cfg_file)
        module = instantiate(cfg.module).load_from_checkpoint(
            ckpt_file,
            backbone=instantiate(cfg.module.backbone),
            criterion=instantiate(cfg.module.criterion),
            optim=instantiate(cfg.module.optim),
        )
        module.eval()

        test_dl = datamodule.predict_dataloader()

        sample_error = dict()
        results = {
            "mse": list(),
            "base_val": list(),
            "maxima_val": list(),
            "maxima_bin": list(),
            "peak_val": list(),
            "peak_bin": list(),
        }
        pbar = tqdm(total=len(test_dl))

        seeds = []
        log.info("dataloader length")
        log.info(len(test_dl))

        log.info("dataset length")
        log.info(len(test_dl.dataset))

        for i, batch in enumerate(test_dl):
            with tr.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                pred_range_profile = module(batch).squeeze()
                batch = {k: v.cpu() for k, v in batch.items()}
                target_profile = batch["data"].cpu().squeeze()
                seeds.append(batch["seed"].cpu().squeeze())

                if i == 0:
                    log.info("targets first")
                    log.info(target_profile[0, :5])
                    log.info("targets last")
                    log.info(target_profile[2, :5])
                    log.info(target_profile.size())

                    log.info("preds first")
                    log.info(pred_range_profile[0, :5])
                    log.info("preds last")
                    log.info(pred_range_profile[2, :5])
                    log.info(pred_range_profile.size())

                if untransform_pred:
                    log.info("converting targets and preds to db")
                    pred = convertLogCenterToDb(
                        pred_range_profile.cpu(),
                        cfg.root_config.std,
                        cfg.root_config.mean,
                    )
                    target = convertLogCenterToDb(
                        target_profile.float(),
                        cfg.root_config.std,
                        cfg.root_config.mean,
                    )
                else:
                    pred = pred_range_profile.cpu()
                    target = target_profile.float()

                # Calculate evlaution metrics
                m_val_diff, m_bin_diff = maxima_matching_score(target, pred, k=2)
                p_val_diff, p_bin_diff = calculated_peak_matching_score(
                    target, pred, max_num_peaks=2
                )
                base_val_diff = calculated_base_matching_score(
                    target, pred, num_low_points=30
                )

                # Store evaluation metrics for batch
                results["mse"].append(tr.mean((target - pred) ** 2))
                results["base_val"].append(base_val_diff)
                results["maxima_val"].append(m_val_diff)
                results["maxima_bin"].append(m_bin_diff)
                results["peak_val"].append(p_val_diff)
                results["peak_bin"].append(p_bin_diff)

                if i == 0:
                    log.info("first batch, stats")
                    for k, v in results.items():
                        msg = f"{k}: {v}"
                        log.info(msg)

                pbar.update(1)

            # Get average metrics over batches
        full_seeds = tr.cat(seeds, dim=0)
        log.info("seeds")
        log.info(full_seeds)

        mse = tr.mean(tr.tensor(results["mse"]))
        base_val = tr.mean(tr.tensor(results["base_val"]))
        maxima_val = tr.mean(tr.tensor(results["maxima_val"]))
        maxima_bin = tr.mean(tr.tensor(results["maxima_bin"]))
        peak_val = tr.mean(tr.tensor(results["peak_val"]))
        peak_bin = tr.mean(tr.tensor(results["peak_bin"]))

        # Print out evaluation metrics
        msg = "MSE: {:.3f} | Base Matching: {:.3f}".format(mse.item(), base_val.item())
        log.info(msg)
        msg = "Maxima Matching Value: {:.3f} | Maxima Matching Bin: {:.3f}".format(
            maxima_val.item(), maxima_bin.item()
        )
        log.info(msg)
        msg = "Peak Matching Value: {:.3f} | Peak Matching Bin: {:.3f}".format(
            peak_val.item(), peak_bin.item()
        )
        log.info(msg)
        # msg = "Val Loss: {:.3f}".format(runs[run_id].data.metrics["val/loss"])
        log.info(msg)

        # Save sample metrics
        if save_metrics_per_sample:
            with open("sample_metrics.pkl", "wb") as fh:
                pickle.dump(sample_error, fh, protocol=pickle.HIGHEST_PROTOCOL)

                saveCsvMetrics(sample_error)


class MeshRTIClassifierEvalWorkflow(_BaseWorkflow):
    @staticmethod
    def task(
        trainer: pl.Trainer,
        datamodule: pl.LightningDataModule,
        mlflow_tracking_uri,
        mlflow_invrt_experiment_num,
        mlflow_invrt_run_id,
    ):

        set_tracking_uri(mlflow_tracking_uri)

        def load_weights(experiment_num, run_id):
            all_runs = MlflowClient().search_runs(
                experiment_ids=str(experiment_num),
            )

            run = [run for run in all_runs if run.info.run_id == run_id][0]
            info = run.info

            output_dir = run.data.params["output_dir"]
            cfg_file = str(list(Path(output_dir).glob("**/config.yaml"))[0])
            ckpt_file = [
                c
                for c in list(Path(output_dir).glob("**/*.ckpt"))
                if "last" not in str(c)
            ][
                0
            ]  # selecting best val error, not last model

            # instantiate model and load weights
            cfg = load_from_yaml(cfg_file)
            module = instantiate(cfg.module).load_from_checkpoint(
                ckpt_file,
                backbone=instantiate(cfg.module.backbone),
                criterion=instantiate(cfg.module.criterion),
                optim=instantiate(cfg.module.optim),
            )
            module.eval()

            return module

        invrt_module = load_weights(mlflow_invrt_experiment_num, mlflow_invrt_run_id)
        invrt_out = trainer.predict(invrt_module, datamodule=datamodule)

        invrt_class_probs = tr.cat([x[2] for x in invrt_out])
        # obtain labels from data loader used for InvRT classification
        labels = tr.cat(
            list(map(lambda x: x[3], list(datamodule.predict_dataloader())))
        )

        # compare
        log.info("Starting metrics")
        log.info(f"labels: {labels}")
        log.info(f"invrt: {invrt_class_probs}")
        # log.info(f'F1 score: {f1_score(labels, invrt_classification)}')
