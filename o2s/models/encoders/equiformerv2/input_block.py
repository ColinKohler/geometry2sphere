import torch
import torch.nn as nn
import copy

from .so3 import SO3_Embedding
from .radial_function import RadialFunction


class EdgeDegreeEmbedding(torch.nn.Module):

    def __init__(
        self,
        sphere_channels,
        
        lmax_list,
        mmax_list,
        
        SO3_rotation,
        mappingReduced,

        edge_channels_list,
        rescale_factor
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        
        self.m_0_num_coefficients = self.mappingReduced.m_size[0] 
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.edge_channels_list = copy.deepcopy(edge_channels_list)


        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = rescale_factor


    def forward(
        self,
        edge_distance,
        edge_index,
        num_nodes
    ):    
        x_edge = edge_distance

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        x_edge_m_pad = torch.zeros((
            x_edge_m_0.shape[0], 
            (self.m_all_num_coefficents - self.m_0_num_coefficients), 
            self.sphere_channels), 
            device=x_edge_m_0.device)
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.sphere_channels, 
            device=x_edge_m_all.device, 
            dtype=x_edge_m_all.dtype
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        # TODO: solve this error index 42 is out of bounds for dimension 0 with size 42
        x_edge_embedding._reduce_edge(edge_index[1] , num_nodes)
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding

