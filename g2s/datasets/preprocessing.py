from typing import Optional, Tuple
import logging

from torch.utils.data import Dataset
import trimesh
import numpy as np
import torch as tr


log = logging.getLogger(__name__)


def _check_viewing_range(
    aspect_angles: Optional[Tuple[float, float]],
    roll_angles: Optional[Tuple[float, float]],
):

    if aspect_angles is not None:
        assert (
            aspect_angles[0] >= -np.pi and aspect_angles[0] <= 2 * np.pi
        ), "Lower bound of aspect angles not in 0 - pi range"
        assert (
            aspect_angles[1] >= -np.pi and aspect_angles[1] <= 2 * np.pi
        ), "Upper bound of aspect angles not in 0 - pi range"

    if roll_angles is not None:
        assert (
            roll_angles[0] >= -np.pi and roll_angles[0] <= 2 * np.pi
        ), "Lower bound of roll angles not in 0 - pi range"
        assert (
            roll_angles[1] >= -np.pi and roll_angles[1] <= 2 * np.pi
        ), "Upper bound of aspect angles not in 0 - pi range"


def restrict_viewing(
    data: Dataset,
    aspect_angles: Optional[Tuple[float, float]] = (np.pi / 4, 3 * np.pi / 4),
    roll_angles: Optional[Tuple[float, float]] = None,
) -> Dataset:
    """Removes all data from the dataset that does not fall in the desired aspect and roll angles. Only works when dataset
        is not in single pulse mode (there must be more than one aspect to process)

    Args:
        data (Dataset): Example to be processed;  should come in the form of an arrow dataset
        aspect_angles (Optional[Tuple[float, float]], optional): Range of acceptable aspect angles in radians. Defaults to (np.pi/4, 3*np.pi/4).
        roll_angles (Optional[Tuple[float, float]], optional): Range of acceptable roll angles in radians. Defaults to None.

    Returns:
        data (Dataset): Updated dataset with reduced viewing angles
    """

    real_data = tr.tensor(data["amp_real"])
    imag_data = tr.tensor(data["amp_imag"])
    aspects = tr.tensor(data["aspects"])
    rolls = tr.tensor(data["rolls"])

    assert (
        aspects.size()[0] != 1 and rolls.size()[0] != 1 and real_data.size()[0] != 1
    ), "This processing function cannot be used on single pulse data; see _single_pulse_restric_viewing fn instead"
    _check_viewing_range(aspect_angles=aspect_angles, roll_angles=roll_angles)

    if aspect_angles is not None:
        aspect_mask = tr.where(
            tr.logical_and(aspects >= aspect_angles[0], aspects <= aspect_angles[1]),
            1,
            0,
        )
    else:
        aspect_mask = tr.ones(data.size())

    if roll_angles is not None:
        rolls_mask = tr.where(
            tr.logical_and(rolls >= roll_angles[0], rolls <= roll_angles[1]), 1, 0
        )
    else:
        rolls_mask = tr.ones(rolls.size())

    aspect_roll_mask = (aspect_mask * rolls_mask).bool()

    masked_real_data = real_data[aspect_roll_mask]
    masked_imag_data = imag_data[aspect_roll_mask]
    masked_aspects = aspects[aspect_roll_mask]
    masked_rolls = rolls[aspect_roll_mask]

    assert (
        masked_real_data.size()[0] > 0
    ), "No data left after filtering; pleae check your aspect and roll bounds"

    data["amp_real"] = masked_real_data.tolist()
    data["amp_imag"] = masked_imag_data.tolist()
    data["aspects"] = masked_aspects.tolist()
    data["rolls"] = masked_rolls.tolist()

    return data


def restrict_viewing_percent(
    data: Dataset,
    percent: float = 0.5,
    restrict_aspect: bool = True,
    restrict_roll: bool = False,
) -> Dataset:
    """Removes viewing angles from data to achieve desired percentage

    Args:
        data (Dataset): Example to be processed;  should come in the form of an arrow dataset
        percent (float): Percentage of viewing angles to remove
        restrict_aspect (bool): If true, retrict viewing angle for aspect
        restrict_roll (bool): If true, retrict viewing angle for roll

    Returns:
        data (Dataset): Updated dataset with reduced viewing angles
    """

    real_data = tr.tensor(data["amp_real"])
    imag_data = tr.tensor(data["amp_imag"])
    aspects = tr.tensor(data["aspects"])
    rolls = tr.tensor(data["rolls"])

    assert percent > 0, "Percent must be greater than 0"
    assert percent <= 1, "Percent must be less than or equal to 0"

    spacing = int(1 / percent)
    unique_rolls = tr.unique(rolls)
    unique_aspects = tr.unique(aspects)

    valid_rolls = unique_rolls[::spacing]
    valid_aspects = unique_aspects[::spacing]

    rolls_mask = tr.isin(rolls, valid_rolls)
    aspects_mask = tr.isin(aspects, valid_aspects)

    if restrict_aspect and restrict_roll:
        aspect_roll_mask = rolls_mask * aspects_mask
    elif restrict_aspect:
        aspect_roll_mask = aspects_mask
    elif restrict_roll:
        aspect_roll_mask = rolls_mask
    else:
        aspect_roll_mask = tr.ones(real_data.size()).bool()

    masked_real_data = real_data[aspect_roll_mask]
    masked_imag_data = imag_data[aspect_roll_mask]
    masked_aspects = aspects[aspect_roll_mask]
    masked_rolls = rolls[aspect_roll_mask]

    assert (
        masked_real_data.size()[0] > 0
    ), "No data left after filtering; pleae check your aspect and roll bounds"

    data["amp_real"] = masked_real_data.tolist()
    data["amp_imag"] = masked_imag_data.tolist()
    data["aspects"] = masked_aspects.tolist()
    data["rolls"] = masked_rolls.tolist()

    return data


def decimate_mesh(data: Dataset, reduction_percent: float = 0.5):

    if reduction_percent < 1:
        mesh = trimesh.Trimesh(data["mesh_vertices"], faces=data["mesh_faces"])
        num_faces = mesh.faces.shape[0]
        target_faces = num_faces // (1 / reduction_percent)
        reduced_mesh = mesh.simplify_quadric_decimation(target_faces)

        original_edges = mesh.edges.shape[0]
        reduced_edges = reduced_mesh.edges.shape[0]

        # msg = f'original_edges: {original_edges}'
        # log.info(msg)
        # msg = f'reduced_edges: {reduced_edges}'
        # log.info(msg)

        data["mesh_faces"] = reduced_mesh.faces.tolist()
        data["mesh_vertices"] = reduced_mesh.vertices.tolist()
        return data
    else:
        return data
