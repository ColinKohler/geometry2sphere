from typing import Any, Dict
import torch

from o2s.datasets.transforms._base import Transform


class Log(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
    ):
        super().__init__()

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return torch.log10(data + 1e-7)

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return 10 ** (data)


class Abs(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
    ):
        super().__init__()

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return data.abs()

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        # print('Not possible to reverse an absolute magnitude transform, so just acting as a pass through')
        return data


class Normalize(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
        min: float,
        max: float,
    ):
        super().__init__()
        self.min = min
        self.max = max

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return (data - self.min) / (self.max - self.min)

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return (data * (self.max - self.min)) + self.min


class Center(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
        mean,
        std,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return (data - self.mean) / self.std

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return (data * self.std) + self.mean
