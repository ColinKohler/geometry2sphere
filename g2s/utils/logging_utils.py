from typing import Any, Dict

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    hparams["module"] = cfg["module"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["train_dataset"] = cfg["train_dataset"]
    hparams["val_dataset"] = cfg["val_dataset"]
    hparams["test_dataset"] = cfg["train_dataset"]
    hparams["trainer"] = cfg["trainer"]
    hparams["batch_size"] = cfg["batch_size"]
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
