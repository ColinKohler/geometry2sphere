import math
from pathlib import Path

import torch as tr
from numpy.random import RandomState

from rai_toolbox.mushin.workflows import multirun
from baselines import configs
from baselines.workflows import TrainingWorkflow

seeds = list(RandomState().randint(0, 100000, size=(10,)))

wf = TrainingWorkflow.run(
    configs.GNNConfig,
    overrides={
        "hydra/launcher": "txg",
        "hydra.sweep.dir": "multirun/training/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "mlflow_tracking_uri": "logging/mlruns",
        "global_seed": multirun(seeds),
        "root_config": "mesh_frustra_v0",
        "trainer.max_epochs": 100,
    },
)
