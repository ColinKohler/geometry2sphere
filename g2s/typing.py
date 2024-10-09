from typing import Optional, TypedDict
from torch.optim.lr_scheduler import _LRScheduler

from rai_toolbox._typing import OptimizerType, Partial


class OptimDict(TypedDict):
    optimizer: OptimizerType
    lr_scheduler: Optional[_LRScheduler]
    frequency: int


class PartialOptimDict(TypedDict):
    optimizer: Partial[OptimizerType]
    lr_scheduler: Optional[Partial[_LRScheduler]]
    frequency: int
