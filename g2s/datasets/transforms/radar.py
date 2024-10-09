from typing import Any, Dict
import torch

from g2s.datasets.transforms._base import Transform


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
        print(
            "Not possible to reverse an absolute magnitude transform, so just acting as a pass through"
        )
        return data


class Normalize(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
        data_min: float,
        data_max: float,
        target_min: float,
        target_max: float,
    ):
        super().__init__()
        self.data_min = data_min
        self.data_max = data_max
        self.target_min = target_min
        self.target_max = target_max

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        r = self.target_max - self.target_min
        norm_data = (
            r * ((data - self.data_min) / (self.data_max - self.data_min))
            + self.target_min
        )
        return norm_data

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        unnorm_data = (data - self.min) * (
            (data.max() - data.min()) / (self.max - self.min)
        ) + data.min()

        return unnorm_data


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


class CenterRangeCrop(Transform):

    AFFECTED_PARAMS = ["data"]

    def __init__(
        self,
        crop_amount: int = 20,
    ):
        super().__init__()
        self.crop_amount = crop_amount

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return data[..., self.crop_amount : -1 * self.crop_amount]

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        print(
            "Not possible to reverse a crop transform, so just acting as a pass through"
        )
        return data
