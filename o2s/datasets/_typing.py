from dataclasses import dataclass
from hydra_zen import MISSING
import torch


@dataclass
class XarrayMeshData:
    seed: torch.Tensor = MISSING
    scale: torch.Tensor = MISSING
    sim_mesh_vertices: torch.Tensor = None
    sim_mesh_faces: torch.Tensor = None
    rep_mesh_vertices: torch.Tensor = MISSING
    rep_mesh_faces: torch.Tensor = MISSING
    frequency: torch.Tensor = None  # TODO: add these back to mesh dataset
    bandwidth: torch.Tensor = None
    relative_range: torch.Tensor = None
    aspect_rad: torch.Tensor = MISSING
    roll_rad: torch.Tensor = MISSING
    data: torch.Tensor = MISSING
    label: torch.Tensor = MISSING


@dataclass
class TransformerXarrayMeshData(XarrayMeshData):
    face_adjacency: torch.Tensor = MISSING
    face_normals: torch.Tensor = MISSING
    centroid: torch.Tensor = MISSING
    triangles_center: torch.Tensor = MISSING
    triangles: torch.Tensor = MISSING
    edges: torch.Tensor = MISSING
