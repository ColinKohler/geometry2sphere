"""
Contains dataset class for mesh datasets for GNNs
"""

from typing import List, Optional, Union, Callable
import os
from pathlib import Path
from numpy import random as npr
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import trimesh
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import xarray as xr
from dataclasses import asdict
from torch.nn.utils.rnn import pad_sequence


from g2s.datasets._typing import TransformerXarrayMeshData
from g2s.datasets.mesh_dataset import MeshXarrayDataset


class RadarDataset(MeshXarrayDataset):
    """Mesh-to-Radar XArray Dataset."""

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
        response = data.data.permute(2, 0, 1).float()
        response = torch.roll(response, response.size(1) // 2, 1)
        response = response[20:-20:1]
        # response = F.interpolate(
        #    response.view(107, 1, 61, 21), scale_factor=4, mode="bilinear"
        # )
        # response = response.squeeze()

        if self.return_mesh:
            sample = (
                mesh,
                response,
                torch.Tensor(data.rep_mesh_vertices),
                torch.Tensor(data.rep_mesh_faces),
            )
        else:
            sample = (mesh, response)

        return sample


class TransformerMeshXarrayDataset(MeshXarrayDataset):

    #   RENAME_DICT = {
    #     # 'aspect_rad':'aspects',
    #     # 'roll_rad':'rolls',
    #     'rep_mesh_vertices':'mesh_vertices',
    #     'rep_mesh_faces':'mesh_faces',

    #   }

    REMOVE_NAMES = [
        "sim_mesh_vertices",
        "sim_mesh_faces",
    ]

    def __init__(
        self,
        root: str,
        stage: str,
        val_size: Optional[
            Union[int, float]
        ] = 0.1,  # use to place additional constraints on the size of the training dataset. Useful for ablations
        train_size: Optional[
            int
        ] = None,  # use to place additional constraints on the size of the training dataset. Useful for ablations
        **kwargs,
    ):
        super().__init__(
            root=root,
            stage=stage,
            val_size=val_size,
            train_size=train_size,
            **kwargs,
        )

    def __getitem__(self, idx, return_mesh: bool = False):
        data = super().__getitem__(idx)

        vertices = data.rep_mesh_vertices
        faces = data.rep_mesh_faces

        assert (
            len(vertices.shape) == 2 and len(faces.shape) == 2
        ), "Vertices and faces arrays are expected to have dimension of 2, but do not have that. Check your underlying data"

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, validate=False, process=False
        )

        # Read the additional information from the mesh
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

        data = asdict(data)

        # for name, new_name in self.RENAME_DICT.items():
        #     data[new_name] = data[name]
        #     del data[name]

        for name in self.REMOVE_NAMES:
            if name in data.keys():
                del data[name]

        output = TransformerXarrayMeshData(
            face_adjacency=face_adjacency,
            face_normals=face_normals,
            centroid=centroid,
            triangles_center=triangles_center,
            triangles=triangles,
            edges=edges,
            **data,
        )

        if return_mesh:
            return output, mesh
        else:
            return output


def radar_transformer_collate_function(
    data: List[torch.Tensor],
    input_id_pad_val: int = 0,
    attention_mask_pad_val: int = 1,
    scaling_tokenizer: Optional[Callable] = None,
    **kwargs,
):

    data = {
        k: [
            (
                asdict(d)[k]
                if type(asdict(d)[k]) == str or torch.is_tensor((asdict(d)[k]))
                else torch.tensor(asdict(d)[k])
            )
            for d in data
        ]
        for k in asdict(data[0]).keys()
        if "sim" not in k
    }
    mesh_attention_mask = pad_sequence(
        [torch.zeros(triangles.shape[0]) for triangles in data["triangles"]],
        batch_first=True,
        padding_value=attention_mask_pad_val,
    ).to(bool)

    aspects = torch.stack(data["aspect_rad"], dim=0)
    rolls = torch.stack(data["roll_rad"], dim=0)

    orientation = torch.stack([aspects, rolls], dim=-1)
    scale = torch.stack(data["scale"], dim=-1)
    if scaling_tokenizer is not None:
        scale = scaling_tokenizer.tokenize(scale)

    collated_data = {
        "data": torch.stack(data["data"], dim=0).squeeze(1),
        # 'label'           :torch.stack(data['label'], dim=0),
        "faces": pad_sequence(
            data["rep_mesh_faces"], batch_first=True, padding_value=input_id_pad_val
        ),
        "vertices": pad_sequence(
            data["rep_mesh_vertices"], batch_first=True, padding_value=input_id_pad_val
        ),
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
        "edges": pad_sequence(
            data["edges"], batch_first=True, padding_value=input_id_pad_val
        ),
        "attention_mask": mesh_attention_mask,
        "orientation": orientation.squeeze(1),
        "scale": scale.squeeze(),
        "seed": torch.stack(data["seed"], dim=0),
    }

    return collated_data
