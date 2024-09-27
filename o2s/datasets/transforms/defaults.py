from typing import Any, Dict
import torch

from o2s.datasets.transforms.radar import Abs, Log, Center, CenterRangeCrop
from o2s.datasets.transforms.general import Compose

#thin wrapper around compose to provide default transform
def get_transformer_transform(
    mean,
    std,
    crop_amount,
    ):
        
    transformer_transform = Compose(
        transforms=[
            Abs(),
            Log(),
            Center(mean=mean, std=std),
            CenterRangeCrop(crop_amount=crop_amount)
        ]
    )

    return transformer_transform
