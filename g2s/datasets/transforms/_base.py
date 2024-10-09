from typing import Any, Callable, Dict, Sequence, Tuple
from abc import abstractmethod, ABC

import torch as tr
from numpy.random import Generator, default_rng
from torch import nn

_TINY = tr.finfo(tr.float32).tiny


class Transform(nn.Module, ABC):

    AFFECTED_PARAMS = []
    DATA_PARAMETER = "data"

    def _check_inputs(self, input: Dict[str, Any]) -> None:
        pass

    def _get_params(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        pass

    def forward(self, input: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        self._check_inputs(input)
        params = self._get_params(input)

        # apply transform only to affected parameters
        if "reverse" in kwargs.keys() and kwargs["reverse"]:
            output = {
                param: (
                    self._untransform(param_val, params)
                    if param in self.AFFECTED_PARAMS
                    else param_val
                )
                for param, param_val in input.items()
            }
        else:
            output = {
                param: (
                    self._transform(param_val, params)
                    if param in self.AFFECTED_PARAMS
                    else param_val
                )
                for param, param_val in input.items()
            }

        if "data_only" in kwargs.keys() and kwargs["data_only"]:
            output = output[self.DATA_PARAMETER]

        return output


class RandomTransform(Transform):

    @abstractmethod
    def _get_params(self, input: Dict[str, Any], rng: Generator) -> Dict[str, Any]:
        return dict()

    def forward(
        self, input: Dict[str, Any], rng: Generator = default_rng(), **kwargs: Any
    ) -> Dict[str, Any]:
        self._check_inputs(input)
        params = self._get_params(input, rng=rng)

        if "reverse" in kwargs.keys() and kwargs["reverse"]:
            output = {
                param: (
                    self._untransform(param_val, params)
                    if param in self.AFFECTED_PARAMS
                    else param_val
                )
                for param, param_val in input.items()
            }
        else:
            output = {
                param: (
                    self._transform(param_val, params)
                    if param in self.AFFECTED_PARAMS
                    else param_val
                )
                for param, param_val in input.items()
            }

        if "data_only" in kwargs.keys() and kwargs["data_only"]:
            output = output[self.DATA_PARAMETER]

        return output
