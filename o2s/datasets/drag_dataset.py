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

from o2s.datasets.mesh_dataset import MeshXarrayDataset


class DragDataset(MeshXarrayDataset):
    """Mesh-to-Drag XArray Dataset."""

    RENAME_DICT = {
        "aspect_rad": "aspect",
        "roll_rad": "rolls",
    }

    def __init__(
        self,
        root: str,
        stage: str,
        mesh_mode: str,
        orientation_mode: str,
        preprocessing_fns: Optional[List[Callable]] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        chunks: Union[int, str] = {"sample": "auto"},
        val_size: Optional[Union[int, float]] = 0.1,
        train_size: Optional[int] = None,
        index_dim: str = "sample",
        label_to_idx: Optional[dict[str, int]] = None,
        return_sim_mesh_data: bool = False,
        return_xr: bool = False,
        padding_value: int = -10,
        upfront_compute: bool = False,
        compute_before_return: bool = True,
        return_mesh: bool = False,
        **kwargs,
    ):
        self.return_mesh = return_mesh

        super().__init__(
            root,
            stage,
            mesh_mode,
            orientation_mode,
            preprocessing_fns,
            transform,
            seed,
            chunks,
            val_size,
            train_size,
            index_dim,
            label_to_idx,
            return_sim_mesh_data,
            return_xr,
            padding_value,
            upfront_compute,
            compute_before_return,
            **kwargs,
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        mesh = trimesh.Trimesh(
            vertices=data.rep_mesh_vertices,
            faces=data.rep_mesh_faces,
            validate=False,
            process=True,
        )
        vertices = torch.tensor(mesh.vertices, dtype=torch.get_default_dtype())
        edges = torch.tensor(mesh.edges_unique).T
        edge_vec = vertices[edges[0]] - vertices[edges[1]]

        non_zero_edge_idx = torch.where(edge_vec.sum(1) != 0)[0]
        non_zero_edges = edges[:, non_zero_edge_idx]
        non_zero_edge_vec = edge_vec[non_zero_edge_idx]

        mesh = Data(
            pos=vertices,
            x=torch.ones(len(vertices), 1),
            edge_index=non_zero_edges,
            edge_vec=non_zero_edge_vec,
        )
        drag = data.data.permute(2, 0, 1)

        if self.return_mesh:
            sample = (
                mesh,
                drag,
                torch.Tensor(data.rep_mesh_vertices),
                torch.Tensor(data.rep_mesh_faces),
            )
        else:
            sample = (mesh, response)

        return sample
