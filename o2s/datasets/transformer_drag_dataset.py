import os
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable
from collections.abc import Iterable
import itertools
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from numpy.random import default_rng
from torch.utils.data import Dataset
from torch_geometric.data import Data

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from sklearn.model_selection import train_test_split
import trimesh

from o2s.datasets.transforms._base import Transform
from o2s.datasets._typing import TransformerXarrayMeshData


log = logging.getLogger(__name__)


class DragDataset(Dataset):

    def __init__(
        self,
        root: str,
        stage: str,
        preprocessing_fns: Optional[List[Callable]] = None,
        transform: Optional[
            Callable
        ] = None,  # TODO define data transformation base class
        seed: Optional[int] = None,
        val_size: Optional[Union[int, float]] = 0.1,
        train_size: Optional[
            int
        ] = None,  # use to place additional constraints on the size of the training dataset. Useful for ablations
        testing=False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.str_data = np.genfromtxt(
            root + "data/case_data.dat", usecols=(0, 1), dtype=str
        )
        self.float_data = np.genfromtxt(
            root + "data/case_data.dat", usecols=(2, 3, 4, 5, 6, 7), dtype=np.float32
        )

        self.preprocessing_fns = preprocessing_fns
        self._set_transforms(transform)

        self.val_size = val_size
        self.stage = stage
        self.seed = seed
        self.root = root
        self.train_size = train_size

        if testing:
            train_idxs = np.arange(self.float_data.shape[0])
            val_idxs = None
            test_idxs = None
        else:
            train_idxs, eval_idxs = train_test_split(
                np.arange(self.float_data.shape[0]),
                train_size=train_size,
                test_size=val_size,
                shuffle=True,
                random_state=seed,
            )
            val_idxs, test_idxs = train_test_split(
                eval_idxs, test_size=0.5, random_state=seed
            )

        self.stage_idxs = {"train": train_idxs, "val": val_idxs, "test": test_idxs}

    def _set_transforms(self, transforms: Union[Callable, Sequence[Callable]]):
        if isinstance(transforms, Iterable):
            self.transforms = transforms
        elif transforms is None:
            self.transforms = None
        else:
            self.transforms = [transforms]

    def __len__(self):
        return len(self.stage_idxs[self.stage])

    def __getitem__(self, idx):
        idx = self.stage_idxs[self.stage][idx]

        # Load mesh
        geom_fp = self.root + "geoms/" + self.str_data[idx, 1] + "_sm.stl"
        try:
            mesh = trimesh.load_mesh(
                geom_fp,
                file_type="stl",
                validate=False,
                process=True,
            )
        except:
            print(geom_fp)

        face_adjacency = torch.tensor(
            mesh.face_adjacency, dtype=torch.get_default_dtype()
        )
        face_normals = torch.tensor(mesh.face_normals, dtype=torch.get_default_dtype())
        centroid = torch.tensor(mesh.centroid, dtype=torch.get_default_dtype())
        triangles_center = torch.tensor(
            mesh.triangles_center, dtype=torch.get_default_dtype()
        )
        triangles = torch.tensor(mesh.triangles, dtype=torch.get_default_dtype())
        edges = torch.tensor(mesh.edges_unique, dtype=torch.get_default_dtype())
        verts = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype())

        # Get node features
        flight_params = torch.from_numpy(self.float_data[idx][:3])
        flight_params[0] = self.norm(flight_params[0], 1, -1, 0, 50)
        flight_params[1] = self.norm(flight_params[1], 1, -1, 1000000, 500000000)
        flight_params[2] = self.norm(flight_params[2], 1, -1, 0, 0.5)

        # Get theta/phi and drag
        out = self.float_data[idx][3:]
        phi = (np.pi / 2.0) - np.radians(out[0])
        theta = np.radians(out[1])
        if theta < 0:
            theta = 2 * np.pi + theta
        coords = torch.tensor([phi, theta]).float()

        # Hacky fix to get the transforms to run on the target
        drag = {"data": torch.tensor(out[2])}
        if self.transforms is not None:
            for d_transform in self.transforms:
                drag = d_transform(drag, rng=default_rng(self.seed))
        drag = drag["data"].reshape(1)

        sample = {
            "face_adjacency": face_adjacency,
            "face_normals": face_normals,
            "centroid": centroid,
            "triangles_center": triangles_center,
            "triangles": triangles,
            "vertices": verts,
            "edges": edges,
            "global_params": flight_params,
            "aspects": coords[0],
            "rolls": coords[1],
            "data": drag,
        }

        return sample

    def norm(self, data, target_max, target_min, data_min, data_max):
        r = target_max - target_min
        norm_data = r * ((data - data_min) / (data_max - data_min)) + target_min

        return norm_data


class NumericTokenizer:
    def __init__(
        self,
        min_val: int,
        max_val: int,
        num_categories: int,
    ):

        self.min_val = min_val
        self.max_val = max_val
        self.num_categories = num_categories
        self.max_bin_num = num_categories - 1

        self.bin_width = (max_val - min_val) / num_categories

    def tokenize(self, values):
        normalized_values = (values - self.min_val) / (self.max_val - self.min_val)
        binned_values = np.clip(
            np.floor(normalized_values * self.num_categories).int(),
            a_min=0,
            a_max=self.max_bin_num,
        )

        return binned_values


def transformer_collate_function(
    data: List[torch.Tensor],
    input_id_pad_val: int = 0,
    attention_mask_pad_val: int = 1,
    **kwargs,
):
    data = {
        k: [
            (
                d[k]
                if type(d[k]) == str or torch.is_tensor((d[k]))
                else torch.tensor(d[k])
            )
            for d in data
        ]
        for k in data[0].keys()
    }
    mesh_attention_mask = pad_sequence(
        [torch.zeros(triangles.shape[0]) for triangles in data["triangles"]],
        batch_first=True,
        padding_value=attention_mask_pad_val,
    ).to(bool)

    aspects = torch.stack(data["aspects"], dim=0)
    rolls = torch.stack(data["rolls"], dim=0)

    orientation = torch.stack([aspects, rolls], dim=-1)

    collated_data = {
        "data": torch.stack(data["data"], dim=0).squeeze(1),
        # "faces": pad_sequence(
        #    data["mesh_faces"], batch_first=True, padding_value=input_id_pad_val
        # ),
        "face_adjacency": pad_sequence(
            data["face_adjacency"], batch_first=True, padding_value=input_id_pad_val
        ),
        "face_normals": pad_sequence(
            data["face_normals"], batch_first=True, padding_value=input_id_pad_val
        ),
        "centroid": pad_sequence(
            data["centroid"], batch_first=True, padding_value=input_id_pad_val
        ),
        "triangles_center": pad_sequence(
            data["triangles_center"], batch_first=True, padding_value=input_id_pad_val
        ),
        "triangles": pad_sequence(
            data["triangles"], batch_first=True, padding_value=input_id_pad_val
        ),
        "vertices": pad_sequence(
            data["vertices"], batch_first=True, padding_value=input_id_pad_val
        ),
        "edges": pad_sequence(
            data["edges"], batch_first=True, padding_value=input_id_pad_val
        ),
        "global_params": torch.stack(data["global_params"], dim=0),
        "attention_mask": mesh_attention_mask,
        "orientation": orientation.squeeze(1),
    }

    return collated_data
