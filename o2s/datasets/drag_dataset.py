import os
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable
from collections.abc import Iterable
import itertools
import random

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
import trimesh

from o2s.datasets.transforms._base import Transform


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

        # Load verts/edges into torch_geometric
        vertices = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype())
        edges = torch.tensor(mesh.edges_unique).T
        edge_vec = vertices[edges[0]] - vertices[edges[1]]

        non_zero_edge_idx = torch.where(edge_vec.sum(1) != 0)[0]
        non_zero_edges = edges[:, non_zero_edge_idx]
        non_zero_edge_vec = edge_vec[non_zero_edge_idx]

        # Get node features
        flight_params = torch.from_numpy(self.float_data[idx][:3])
        flight_params[0] = self.norm(flight_params[0], 1, -1, 0, 50)
        flight_params[1] = self.norm(flight_params[1], 1, -1, 1000000, 500000000)
        flight_params[2] = self.norm(flight_params[2], 1, -1, 0, 0.5)

        mesh = Data(
            pos=vertices,
            x=torch.ones(len(vertices), 1),
            edge_index=non_zero_edges,
            edge_vec=non_zero_edge_vec,
        )

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

        sample = (mesh, flight_params, coords, drag)
        return sample

    def norm(self, data, target_max, target_min, data_min, data_max):
        r = target_max - target_min
        norm_data = r * ((data - data_min) / (data_max - data_min)) + target_min

        return norm_data
