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

import xarray as xr
from sklearn.model_selection import train_test_split
import trimesh

from o2s.datasets._typing import XarrayMeshData
from o2s.datasets.transforms._base import Transform


log = logging.getLogger(__name__)


class MeshXarrayDataset(Dataset):

    AVAILABLE_MESH_RETURN_MODES = ["full", "simple", "iterative", "ordered_iterative"]
    AVAILABLE_ORIENTATION_RETURN_MODES = [
        "full",
        "full_flattened",
        "single",
        "single_flattened",
    ]

    def __init__(
        self,
        root: str,
        stage: str,
        mesh_mode: str,
        orientation_mode: str,
        preprocessing_fns: Optional[List[Callable]] = None,
        transform: Optional[
            Callable
        ] = None,  # TODO define data transformation base class
        seed: Optional[int] = None,
        # chunks:Union[int, str]='auto',
        chunks: Union[int, str] = {"sample": "auto"},
        val_size: Optional[Union[int, float]] = 0.1,
        train_size: Optional[
            int
        ] = None,  # use to place additional constraints on the size of the training dataset. Useful for ablations
        index_dim: str = "sample",
        label_to_idx: Optional[dict[str, int]] = None,
        return_sim_mesh_data: bool = False,
        return_xr: bool = False,
        padding_value: int = -10,
        upfront_compute: bool = False,
        compute_before_return: bool = True,
        **kwargs,
    ):
        super().__init__()
        if os.path.isfile(str(root)):
            self.dataset = xr.open_dataset(root, engine="netcdf4")
        else:  # assume directory
            paths = [Path(root) / p for p in os.listdir(root) if ".nc" in p]
            self.dataset = xr.open_mfdataset(
                paths, chunks=chunks, concat_dim=index_dim, combine="nested"
            )  # todo - add additional speed up techniques here

        if upfront_compute:
            print("computing")
            self.dataset = self.dataset.compute()

        self.upfront_compute = upfront_compute
        self.compute_before_return = compute_before_return

        self.preprocessing_fns = preprocessing_fns

        self._set_transforms(transform)

        self.val_size = val_size
        self.stage = stage
        self.seed = seed
        self.root = root
        self.train_size = train_size
        self.index_dim = index_dim
        self.padding_value = padding_value
        self.return_sim_mesh_data = return_sim_mesh_data
        self.return_xr = return_xr
        if chunks == "auto":
            self.chunks = {index_dim: "auto"}
        else:
            self.chunks = chunks

        if label_to_idx is None:
            class_names = sorted(np.unique(self.dataset.label.values))
            self.label_to_idx = {k: i for i, k in enumerate(class_names)}

        assert (
            mesh_mode in self.AVAILABLE_MESH_RETURN_MODES
        ), "Unsupported mesh mode selected. Please choose another mode"
        self.mesh_mode = mesh_mode
        if "iterative" in self.mesh_mode:
            self.rep_mesh_per_idx = self.dataset.isel(
                {self.index_dim: 0}
            ).rep_mesh_vertices.mesh_variant.size

        assert (
            orientation_mode in self.AVAILABLE_ORIENTATION_RETURN_MODES
        ), "Unsupported orientation mode selected. Please choose another mode"
        self.orientation_mode = orientation_mode
        self.roll_idx = None
        if "single" in self.orientation_mode:
            if "roll_idx" in kwargs.keys():
                self.roll_idx = kwargs["roll_idx"]
            elif "roll_val" in kwargs.keys():
                raise NotImplementedError("Need to add this functionality")
            else:
                self.roll_idx = 0

        self._split_dataset()

    def _set_transforms(self, transforms: Union[Callable, Sequence[Callable]]):

        if isinstance(transforms, Iterable):
            self.transforms = transforms
        elif transforms is None:
            self.transforms = None
        else:
            self.transforms = [transforms]

    def _split_dataset(self):
        base_mesh_idxs = list(range(len(self.dataset[self.index_dim])))
        if self.stage == "test" or self.val_size is None:
            self.idxs = base_mesh_idxs

            if "iterative" in self.mesh_mode:
                if "ordered" not in self.mesh_mode:
                    random.Random(self.seed).shuffle(base_mesh_idxs)

                self.idxs = list(
                    itertools.chain(
                        *[
                            (
                                np.arange(0, self.rep_mesh_per_idx)
                                + base_idx * self.rep_mesh_per_idx
                            ).tolist()
                            for base_idx in base_mesh_idxs
                        ]
                    )
                )

        elif self.stage == "train" or self.stage == "val":
            # in these full and simple modes, only one sample per index, so the length of the index dimension is the length of the dataset
            train_idxs, val_idxs = train_test_split(
                base_mesh_idxs,
                test_size=self.val_size,
                shuffle=True,
                train_size=self.train_size,
                random_state=self.seed,
            )

            if "iterative" in self.mesh_mode:
                # in this mode each representation mesh is treated as a seperate sample, so the total dataset size is the number of
                # rep meshes multiplied by the length of the dataset

                # if ordered iterative, do not shuffle
                train_idxs = list(
                    itertools.chain(
                        *[
                            (
                                np.arange(0, self.rep_mesh_per_idx)
                                + base_idx * self.rep_mesh_per_idx
                            ).tolist()
                            for base_idx in train_idxs
                        ]
                    )
                )
                val_idxs = list(
                    itertools.chain(
                        *[
                            (
                                np.arange(0, self.rep_mesh_per_idx)
                                + base_idx * self.rep_mesh_per_idx
                            ).tolist()
                            for base_idx in val_idxs
                        ]
                    )
                )

                if self.mesh_mode == "iterative":
                    random.Random(self.seed).shuffle(train_idxs)
                    random.Random(self.seed).shuffle(val_idxs)

                # want to make sure that all models in val don't have any mesh variants in the training set, since other wise it would be trivial to match the val
            if self.stage == "train":
                self.idxs = train_idxs
            else:  # val stage
                self.idxs = val_idxs
        else:
            raise ValueError('Stage must be set to either "train", "val", "test"')

    def _unpad_mesh_data(self, input):
        # note - assumes that padding value does not occur simulatenously in all the xyz locations at any point in the dataset
        # this is a fine assumption to make if meshes are normalized (can just set padding value outside of data range) but may cause issues in other circumstances
        assert (
            len(input.shape) == 2
        ), "Unpad mesh currently only supports 2 dimensional inputs (no batch processing)"
        assert input.shape[1] == 3, "Second dimension must be vertex/face dimension"
        if self.padding_value is not None:
            mask = np.all(np.array(input) == self.padding_value, axis=-1)
            if np.all(mask):
                return input  # no placeholder rows
            else:
                num_placeholders = np.sum(mask)
            return input[:-num_placeholders, :]

    def _test_base_mesh_idxs(self):
        base_idxs = []
        mesh_rep_idxs = []
        for idx in self.idxs:
            mesh_rep_idxs.append(idx % self.rep_mesh_per_idx)
            base_idxs.append(idx // self.rep_mesh_per_idx)

        return base_idxs, mesh_rep_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # print(idx)
        idx = self.idxs[idx]
        # print(idx)
        if "iterative" in self.mesh_mode:
            mesh_rep_idx = idx % self.rep_mesh_per_idx
            idx = idx // self.rep_mesh_per_idx
            # print(idx, mesh_rep_idx)
            # print()

        elif self.mesh_mode == "simple":
            mesh_rep_idx = 0
        else:
            mesh_rep_idx = None

        data = {}
        sample_data = self.dataset.isel({self.index_dim: idx})
        if self.return_xr:
            if self.compute_before_return:
                sample_data = sample_data.compute()
            return sample_data

        if self.return_sim_mesh_data:
            data["sim_mesh_vertices"] = self._unpad_mesh_data(
                sample_data.sim_mesh_vertices.values
            )
            data["sim_mesh_faces"] = self._unpad_mesh_data(
                sample_data.sim_mesh_faces.values
            )

        rep_mesh_vertices = torch.tensor(sample_data.rep_mesh_vertices.values)
        rep_mesh_faces = torch.tensor(sample_data.rep_mesh_faces.values)

        if mesh_rep_idx is not None:
            rep_mesh_vertices = self._unpad_mesh_data(
                rep_mesh_vertices[mesh_rep_idx, ...]
            )
            rep_mesh_faces = self._unpad_mesh_data(rep_mesh_faces[mesh_rep_idx, ...])

        # Note - if not in iterative mode these meshes are still padded.  Downstream datasets must unpad before use
        data["rep_mesh_vertices"] = rep_mesh_vertices
        data["rep_mesh_faces"] = rep_mesh_faces

        # metadata
        data["seed"] = sample_data.seed.values.item()
        data["scale"] = sample_data.scale.values.item()
        data["frequency"] = sample_data.frequency.values.item()
        data["bandwidth"] = sample_data.bandwidth.values.item()
        data["label"] = self.label_to_idx[sample_data.label.values.item()]

        data["relative_range"] = torch.tensor(
            self.dataset.range.values, dtype=torch.get_default_dtype()
        )
        data["aspect_rad"] = torch.tensor(
            self.dataset.aspect_rad.values, dtype=torch.get_default_dtype()
        )
        data["roll_rad"] = torch.tensor(
            self.dataset.roll_rad.values, dtype=torch.get_default_dtype()
        )
        # data["data"] = torch.view_as_complex(
        #    torch.tensor(sample_data["rti"].values, dtype=torch.get_default_dtype())
        # )
        data["data"] = torch.tensor(
            sample_data["rti"].values, dtype=torch.get_default_dtype()
        )[:, :, :, 0]

        if "single" in self.orientation_mode:
            data["data"] = data["data"][:, self.roll_idx, :]
            data["roll_rad"] = data["roll_rad"][self.roll_idx]
            if "flatten" in self.orientation_mode:
                num_aspects = len(data["aspect_rad"])
                data["roll_rad"] = torch.tensor(
                    num_aspects * [data["roll_rad"].item()],
                    dtype=torch.get_default_dtype(),
                )
        elif self.orientation_mode == "full_flattened":
            num_rolls = len(data["roll_rad"])
            num_aspects = len(data["aspect_rad"])
            data["aspect_rad"] = torch.tile(data["aspect_rad"], (num_rolls,))
            data["roll_rad"] = torch.repeat_interleave(data["roll_rad"], num_aspects)

        if self.transforms is not None:
            for d_transform in self.transforms:
                data = d_transform(data, rng=default_rng(self.seed))

        output = XarrayMeshData(**data)

        return output
