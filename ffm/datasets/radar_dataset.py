"""
Contains dataset class for mesh datasets for GNNs
"""

from typing import List, Optional, Union, Callable
import os
from pathlib import Path
from numpy import random as npr
import torch
from torch_geometric.data import Data
import trimesh
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import xarray as xr


class RadarDataset(Dataset):
    """Mesh-to-Radar XArray Dataset."""

    RENAME_DICT = {
        "aspect_rad": "aspect",
        "roll_rad": "rolls",
    }

    def __init__(
        self,
        root: str,
        stage: str,
        preprocessing_fns: Optional[List[Callable]] = None,
        transform: Callable = None,
        val_size: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        chunks: Union[int, str] = "auto",
        return_mesh: bool = False,
    ):
        super().__init__()
        if os.path.isfile(str(root)):
            self.dataset = xr.open_dataset(root, engine="netcdf4")
        else:
            paths = [Path(root) / p for p in os.listdir(root) if ".nc" in p]
            self.dataset = xr.open_mfdataset(
                paths, chunks=chunks, concat_dim="sample", combine="nested"
            )
            for field_name in self.RENAME_DICT.keys():
                if field_name in (self.dataset.coords.keys()):
                    self.dataset = self.dataset.rename(
                        {field_name: self.RENAME_DICT[field_name]}
                    )

        self.root = root
        self.preprocessing_fns = preprocessing_fns
        self.transform = transform
        self.val_size = val_size
        self.stage = stage
        self.seed = seed
        self.data_dict = None
        self.coords = None
        self.return_mesh = return_mesh

        self._split_dataset()

    def _split_dataset(self):
        if self.stage == "test" or self.val_size is None:
            # assumes that test set should not be split and ignores params
            self.data_dict = {
                x: y.to_numpy() for x, y in dict(self.dataset.data_vars).items()
            }
            self.coords = dict(self.dataset.coords)
        elif self.stage in ["train", "val"]:
            train_datavars = {}
            test_datavars = {}
            train_coords = {}
            test_coords = {}

            for key, val in dict(self.dataset.data_vars).items():
                train_datavars[key], test_datavars[key] = train_test_split(
                    self.dataset[key].to_numpy(), test_size=self.val_size, shuffle=False
                )

            for key, val in dict(self.dataset.coords).items():
                train_coords[key] = self.dataset.coords[key]
                test_coords[key] = self.dataset.coords[key]

            if self.stage == "train":
                self.data_dict = train_datavars
                self.coords = train_coords
            else:
                self.data_dict = test_datavars
                self.coords = test_coords
        else:
            raise ValueError('Stage must be set to either "train", "val", "test"')

    def __getitem__(self, idx):
        samp = {x: self.data_dict[x][idx] for x in self.data_dict}
        mesh = trimesh.Trimesh(
            vertices=samp["mesh_vertices"],
            faces=samp["mesh_faces"],
            validate=False,
            process=False,
        )
        vertices = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype())
        edges = torch.tensor(mesh.edges).T
        edge_vec = vertices[edges[0]] - vertices[edges[1]]
        mesh = Data(
            pos=vertices,
            x=torch.ones(len(vertices), 1),
            edge_index=edges,
            edge_vec=edge_vec,
        )

        response = torch.view_as_complex(
            torch.stack(
                [
                    torch.tensor(samp["rti"])[:, :, :, 0],
                    torch.tensor(samp["rti"])[:, :, :, 1],
                ]
            )
            .permute(3, 1, 2, 0)
            .contiguous()
        )
        if self.transform is not None:
            if self.seed is not None:
                rng = npr.default_rng([self.seed, abs(idx)])
            else:
                rng = npr.default_rng()

            response = self.transform(rng=rng, data=response)["data"]

        if self.return_mesh:
            sample = (
                mesh,
                response,
                torch.Tensor(samp["mesh_vertices"]),
                torch.Tensor(samp["mesh_faces"]),
            )
        else:
            sample = (mesh, response)

        return sample

    def __len__(self):
        return len(self.data_dict["mesh_vertices"])
