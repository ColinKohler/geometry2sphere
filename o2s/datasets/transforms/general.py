from typing import Any, Dict, List, Optional
from numpy.random import Generator
import torch
from torch import nn
from o2s.datasets.transforms._base import Transform


class Compose(nn.Module):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        assert len(transforms) > 0, "At least one transform must be included in compose"
        self.transforms = transforms

    def _transform(
        self, input: Dict[str, Any], rng: Optional[Generator] = None
    ) -> Dict[str, Any]:
        updated_input = input
        for transform in self.transforms:
            updated_input = transform(input=updated_input, rng=rng)

        return updated_input

    def _untransform(
        self, input: Dict[str, Any], rng: Optional[Generator] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        transforms = list(reversed(self.transforms))
        updated_input = input
        for transform in transforms:
            updated_input = transform(input=updated_input, rng=rng, reverse=True)

        return updated_input

    def forward(
        self, input: Dict[str, Any], rng: Optional[Generator] = None, **kwargs: Any
    ) -> torch.Tensor:
        if "reverse" in kwargs.keys() and kwargs["reverse"]:
            output = self._untransform(input=input, rng=rng)
        else:
            output = self._transform(input=input, rng=rng)

        if "data_only" in kwargs.keys() and kwargs["data_only"]:
            output = output[self.DATA_PARAMETER]

        return output


class DummyTransform(Transform):
    def __init__(
        self,
    ):
        super().__init__()

    def _transform(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return input

    def _untransform(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return input
