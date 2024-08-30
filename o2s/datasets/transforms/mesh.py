from typing import Any, Dict, Optional, Sequence, Tuple, List

from o2s.datasets.transforms._base import Transform


class MeshNormalize(Transform):

    AFFECTED_PARAMS = ["rep_mesh_vertices"]

    def _check_inputs(self, input: Dict[str, Any]) -> None:
        assert (
            "scale" in input.keys()
        ), "Mesh scaling not implemented when scale not provided"

    def _get_params(self, input: Dict[str, Any]) -> Dict[str, Any]:
        scaling_factor = (
            input["scale"] / 2.0
        )  # since scale measures negative to positive centered at 0, this gets us to the unit cube
        return {"scaling_factor": scaling_factor}

    def _transform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return data / params["scaling_factor"]

    def _untransform(
        self, data: Any, params: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return data * params["scaling_factor"]
