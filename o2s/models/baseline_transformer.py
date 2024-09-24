from __future__ import annotations

from typing import Any, Callable, List, Optional, Type, Union, Dict
from abc import ABC, abstractmethod
import os

import torch
import torch as tr
import numpy as np
from torch import Tensor, nn
from torch.nn.functional import interpolate

from pathlib import Path

def get_expansion_layer(
    initial_embedding_size:int,
    target_embedding_size:int,
    expansion_layers:int = 1,
    residual_layers:int = 0,
):
    feature_diff = target_embedding_size - initial_embedding_size
    feature_step_size = feature_diff//expansion_layers
    expansion_layers = []
    current_size = initial_embedding_size
    while current_size < target_embedding_size - feature_step_size:
        expansion_layers.append(nn.Linear(current_size, current_size + feature_step_size))
        expansion_layers.append(nn.ReLU(inplace=True))
        current_size += feature_step_size
    expansion_layers.append(nn.Linear(current_size, target_embedding_size))

    expansion_layers = nn.Sequential(*expansion_layers)
    if residual_layers != 0:
        residual_layers = nn.Sequential(*[FFResidualBlock(target_embedding_size) for _ in range(residual_layers)])
        return nn.Sequential(expansion_layers, residual_layers)
    else:
        return expansion_layers

class FFResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int, 
        ):
        super().__init__()
        self.dim = dim
        self.ff1 = nn.Linear(dim, dim)
        self.ff2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)


    def forward(
        self, 
        encoding: Tensor
        ):
        identity = encoding

        out = self.ff1(encoding)
        out = self.relu(out)
        out = self.ff2(out)
        out += identity

        return out

class TransformerDecoderBase(nn.Module):
    def __init__(
        self,
        dim: int, 
        ):
        super().__init__()
        self.dim=dim

    @abstractmethod
    def forward(
        self, 
        encoding: Tensor #batch size x sequence length x hidden size
        ):
        """
        Takes a transformer encoding and outputs a simulation output
        """
        pass


class RadarTransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        dim: int, 
        num_range_gates: int,
        num_viewing_geometries: int = 1,
        internal_num_range_gates: Optional[int] = None,
        internal_num_viewing_geometries: Optional[int] = None,
        interpolation_mode: Optional[str]='bilinear'
        ):
        super().__init__(dim=dim)
        self.num_range_gates=num_range_gates
        self.num_viewing_geometries=num_viewing_geometries
        if num_range_gates == internal_num_range_gates or internal_num_range_gates is None:
            self.internal_num_range_gates=num_range_gates
        else:
            self.internal_num_range_gates=internal_num_range_gates

        if num_viewing_geometries == internal_num_viewing_geometries or internal_num_viewing_geometries is None:
            self.internal_num_viewing_geometries=num_viewing_geometries
        else:
            self.internal_num_viewing_geometries=internal_num_viewing_geometries

        self.interpolation_mode=interpolation_mode

        self.encoding_residual = FFResidualBlock(dim)
        effective_output_dim = self.internal_num_viewing_geometries*self.internal_num_range_gates
        self.scale = nn.Linear(dim, effective_output_dim)
        self.output_residual = FFResidualBlock(internal_num_range_gates) #can't handle very large output dimensions, so only do over the range gate dimension
    
    def forward(
        self, 
        cls_encoding: Tensor,
        return_raw_output: bool = False,
        ):
        """
        Currently predicts only the magnitude - fine for initial work
        """
        # cls_encoding = encoding[:,0,:]
        cls_encoding = self.encoding_residual(cls_encoding)
        cls_encoding = self.scale(cls_encoding)
        range_predictions = cls_encoding.reshape((-1, 1, self.internal_num_viewing_geometries, self.internal_num_range_gates)) #batch x channels (1) x viewing geometries x range gates

        range_predictions = self.output_residual(range_predictions)

        if not return_raw_output:
            range_predictions = interpolate(range_predictions, size=(self.num_viewing_geometries,self.num_range_gates), mode=self.interpolation_mode)

        return range_predictions[:, 0, :, :] #remove channel dimension before returning


class DragTransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        dim: int, 
        num_global_parameters: int,
        num_output_predictions: int
        ):
        super().__init__(dim=dim)
        self.num_global_parameters = num_global_parameters
        self.num_output_parameters = num_output_predictions

        self.encoding_residual = FFResidualBlock(dim)
        self.output_layer = nn.Linear(dim+num_global_parameters, num_output_predictions)

    def forward(
        self, 
        cls_encoding: Tensor,
        global_params: Tensor,
        ):
        """
        Currently predicts only the magnitude - fine for initial work
        """
        # cls_encoding = encoding[:,0,:]
        cls_encoding = self.encoding_residual(cls_encoding)
        cls_encoding = torch.cat([cls_encoding, global_params], dim=-1)

        drag_prediction = self.output_layer(cls_encoding)

        return drag_prediction


class MeshSpatialEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int = 64,
        expansion_layers: int = 1,
        residual_layers: int = 0,
        ):
        super().__init__()
        """Default values based on Meshnet paper"""

        self.embedding_size = embedding_size
        self.expansion_layers = expansion_layers
        self.residual_layers = residual_layers

        self.embedding_layer = get_expansion_layer(3, embedding_size, expansion_layers, residual_layers)

    def forward(
        self, 
        triangles_center: Tensor #mesh_info.triangles_center
        ):
        assert len(triangles_center.shape) == 3, 'Dimensions of triangle centers should be batch x num faces x 3'

        spatial_embedding = self.embedding_layer(triangles_center)

        return spatial_embedding   

class MeshStructuralEmbeddingLayer(nn.Module):
    def __init__(
        self,
        cv_embed_size: int = 64,
        struct_embed_size: int = 64,
        cv_expansion_layers: int = 1,
        cv_residual_layers: int = 0, 
        struct_expansion_layers: int = 1,
        struct_residual_layers: int = 0, 
        ):
        """Default values based on Meshnet paper"""
        super().__init__()
        self.face_kernels = get_expansion_layer(6, cv_embed_size, cv_expansion_layers, cv_residual_layers)
        self.normal_embedding_layer = get_expansion_layer(3, cv_embed_size, cv_expansion_layers, cv_residual_layers)
        self.embedding_layer = get_expansion_layer(2*cv_embed_size, struct_embed_size, struct_expansion_layers, struct_residual_layers)


    #TODO: add additional information on difference between face normals and adjacent normals (or rather curvature information)
    def forward(
        self, 
        face_normals: Tensor, #mesh.face_normals
        triangles: Tensor, #mesh_info.triangles
        triangles_center: Tensor, #mesh_info.triangles_center
        ):

        assert len(face_normals.shape) == 3, 'Dimensions of triangle centers should be batch x num faces x 3'
        assert len(triangles.shape) == 4, 'Dimensions of triangle centers should be batch x num faces x 3 x 3'
        assert len(triangles_center.shape) == 3, 'Dimensions of triangle centers should be batch x num faces x 3'
   
        #get corner vectors
        corner_vectors = triangles - triangles_center[:, :, None, :] #need to double check the order of dimensions
        
        #create corner vector pairs
        cv_0 = corner_vectors[:, :, 0, :]
        cv_1 = corner_vectors[:, :, 1, :]
        cv_2 = corner_vectors[:, :, 2, :]
        cv_pair_0 = torch.cat([cv_0, cv_1], dim=-1)
        cv_pair_1 = torch.cat([cv_1, cv_2], dim=-1)
        cv_pair_2 = torch.cat([cv_2, cv_0], dim=-1)

        #apply corner vector kernels
        cv_0_embed = self.face_kernels(cv_pair_0)
        cv_1_embed = self.face_kernels(cv_pair_1)
        cv_2_embed = self.face_kernels(cv_pair_2)

        #average pool corner vector embeddings to remove ordering information
        cv_embed = (1/3)*(cv_0_embed + cv_1_embed + cv_2_embed) 

        #generate normal embed
        normal_embedding = self.normal_embedding_layer(face_normals)

        #concatenate structural embeddings
        struct_embedding = torch.cat([normal_embedding, cv_embed], dim=-1)

        # generate final embedding
        struct_embedding = self.embedding_layer(struct_embedding)

        return struct_embedding

class MeshEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        spatial_embed_size: int = 64,
        cv_embed_size: int = 64,
        struct_embed_size: int = 64,
        expansion_layers: int = 1,
        residual_layers: int = 0,
        scaling_layer: CategoricalEmbedding|None = None,
        ):
        super().__init__()
        """Default values based on Meshnet paper"""
        self.spatial_embed_size = spatial_embed_size
        self.cv_embed_size = cv_embed_size
        self.struct_embed_size = struct_embed_size
        self.embedding_size = embedding_size

        self.expansion_layers = expansion_layers
        self.residual_layers = residual_layers

        self.spatial_embedder = MeshSpatialEmbeddingLayer(
            embedding_size=spatial_embed_size,
            expansion_layers=expansion_layers,
            residual_layers=residual_layers
        )

        self.structural_embedder = MeshStructuralEmbeddingLayer(
            cv_embed_size=cv_embed_size,
            struct_embed_size=struct_embed_size,
            cv_expansion_layers=expansion_layers,
            cv_residual_layers=residual_layers,
            struct_expansion_layers=expansion_layers,
            struct_residual_layers=residual_layers
        )

        self.embedding_layer = get_expansion_layer(spatial_embed_size+struct_embed_size, embedding_size, expansion_layers, residual_layers)
        self.cls_layer = CLSEmbedding(embedding_size)

        self.scaling_layer = scaling_layer

    def forward(
        self, 
        data: Dict
        ):
        
        #get spatial and structural embeddings
        spatial_embedding = self.spatial_embedder(data['triangles_center'])

        structural_embedding = self.structural_embedder(
            data['face_normals'], 
            data['triangles'], 
            data['triangles_center']
        )

        #concatenate spatial and structural embeddings
        embedding = torch.cat([spatial_embedding, structural_embedding], dim=-1)

        #process embeddings to reach final size
        embedding = self.embedding_layer(embedding)

        #add scaling embedding
        if self.scaling_layer is not None:
            embedding = self.scaling_layer(embedding, data['scale'])

        # add cls embedding
        embedding = self.cls_layer(embedding)

        #update attention vector to include cls embedding and orientation embedding
        if self.scaling_layer is not None:
            attention_expansion = 2
        else:
            attention_expansion = 1

        attention_mask = data['attention_mask']
        expanded_attention_mask = torch.zeros(attention_mask.shape[0], attention_mask.shape[1] + attention_expansion).bool().to(attention_mask.device)
        expanded_attention_mask[:, attention_expansion:] = attention_mask

        return embedding, expanded_attention_mask  

class CategoricalEmbedding(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_categories: int
    ) -> None:
        """
        Additional embedding for categorical information

        Parameters
        ----------
        embed_size : int
            Size of embedding. Must match the size of the other encoders.
        num_categories: int
            number of categories to generate embeddings for 
        """
        super().__init__()
        self.embed_size = embed_size
        self.num_categories = num_categories

        self.embed_layer = torch.nn.Embedding(embedding_dim=embed_size, num_embeddings=num_categories)

    def forward(self, x: Tensor, info:Tensor) -> Tensor:
        """
        Generate embedding vector and add to the front of embeded vector

        Parameters
        ----------
        x : Tensor, shape-(N, R)
            `R` is the dimension of the range profile and `N` are the number of other embeddings.
        info Tensor, shape-(1, N)
            `N` is the batch size. Must contain only int tensors or long tensors

        Returns
        -------
        Tensor, shape-(N, embed size)
            Returns embedding of the categorical information by pulling the information from the embedding dictionary.
        """
        if len(x.shape) == 2: #if concatenating with cls embedding
            info_embed = self.embed_layer(info)
            x = torch.cat([info_embed, x], dim=-1)
        else:
            info_embed = self.embed_layer(info)
            x = torch.cat([info_embed[:,None, :], x], dim=1)

        return x

class CLSEmbedding(nn.Module):
    def __init__(
        self,
        embed_size: int,
    ) -> None:
        """
        Additional class (CLS) embedding. Neccessary for extracting representations from data.

        Parameters
        ----------
        embed_size : int
            Size of embedding. Must match the size of the other encoders.
        """
        super().__init__()
        self.cls_embed = nn.Parameter(torch.rand(1, embed_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Add cls embedding to the first position of an embedding.

        Parameters
        ----------
        x : Tensor, shape-(N, R)
            `R` is the dimension of the range profile and `N` are the number of other embeddings.

        Returns
        -------
        Tensor, shape-(N, 1 + R)
            Returns original tensor with the classification token appended to the front of the tensor.
        """
        N, _ = x.shape[0], x.shape[1]
        x = torch.cat([self.cls_embed.expand(N, -1)[:, None, :], x], dim=1)

        return x

class MeshTransformerSimulator(nn.Module):
    """
    Transformer architecture for converting input shape and aspect to radar range profile
    """
    def __init__(self,
    mesh_embedder,
    n_layers: int, 
    n_heads: int,
    num_range_gates: int,
    num_viewing_geometries:int=1,
    internal_num_range_gates:Optional[int] = None,
    internal_num_viewing_geometries:Optional[int] = None,
    dropout=0.5,
    decoder_arch: str = 'radar',
    mesh_encoder: None|nn.Module=None,
    scaling_layer:None|CategoricalEmbedding=None,
    **kwargs
    ) -> None:
        super().__init__()


        self.embedder = mesh_embedder

        if hasattr(mesh_embedder, 'model'): #stand in for type checking if mehgpt encoder or not. THese cause issues with DDP and arent'y needed, so remove
            self.embedder.model.decoders = None 
            self.embedder.model.init_decoder_conv = None
            self.embedder.model.to_coord_logits = None
            if self.embedder.use_encoding:
                self.embedder.model.project_dim_codebook = None

        d_model = mesh_embedder.embedding_size
        
        #have to use this instead of hugging face models since they accept embeddings directly
        if mesh_encoder is not None:
            print('using provided mesh encoder')
            self.encoder = mesh_encoder
            self.encoding_size = mesh_encoder.encoding_size


        else:
            print('creating new mesh encoder')
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=n_heads * d_model,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=n_layers,
                norm=nn.LayerNorm(d_model, eps=1e-5),
            )
            self.encoding_size = d_model
            
            
        self.scaling_layer = scaling_layer

        #if using pretrained models, there can be a mismatch
        self.resize_layer = None
        if mesh_embedder.embedding_size != self.encoding_size:
            self.resize_layer = tr.nn.Linear(mesh_embedder.embedding_size, self.encoding_size)

        #add other decoder options here as they are created
        if decoder_arch == 'radar':
            if self.scaling_layer is not None:
                embed_size = self.encoding_size + scaling_layer.embed_size
            else:
                embed_size = self.encoding_size
            
            self.decoder = RadarTransformerDecoder(
                dim=embed_size, 
                num_viewing_geometries=num_viewing_geometries, 
                num_range_gates=num_range_gates, 
                internal_num_range_gates=internal_num_range_gates,
                internal_num_viewing_geometries=internal_num_viewing_geometries
                )
            self.decoder_type = 'radar'
        elif decoder_arch == 'drag':

            if self.scaling_layer is not None:
                embed_size = self.encoding_size + scaling_layer.embed_size
            else:
                embed_size = self.encoding_size
                
            self.decoder = DragTransformerDecoder(
                dim=embed_size, 
                num_global_parameters=kwargs['drag_global_parameters'],
                num_output_predictions=kwargs['drag_prediction_parameters'],
            )
            self.decoder_type = 'drag'
            
        else:
            raise ValueError('Invalid option provided for decoder architecture')

        
    def forward(self, data, *args, **kwargs):
        embedding, updated_attention_mask = self.embedder(data)
        if self.resize_layer is not None:
            embedding = self.resize_layer(embedding)

        full_encoding = self.encoder(embedding, src_key_padding_mask=updated_attention_mask)
        cls_encoding = full_encoding[:,0,:]
        if self.scaling_layer is not None:
            cls_encoding = self.scaling_layer(cls_encoding, data['scale'])
        if self.decoder_type == 'radar':
            range_profile = self.decoder(cls_encoding, **kwargs)
            if 'return_embed' in kwargs and kwargs['return_embed']:
                cls_embedding = full_encoding[:,0,:]
                return range_profile, cls_embedding
            else:
                return range_profile
        elif self.decoder_type == 'drag':
            drag_preds = self.decoder(cls_encoding, data['global_params'])
            return drag_preds
        else:
            raise NotImplementedError
        
    #not handling range gate interp right now, as that has an actual physical meaning
    def getResponse(
            self, 
            data:dict, #assumes this is given in batch form after running through collate function
            orientations #assume that this is in radians and is in form N x time x (aspect/roll)
            ):
        assert len(data.size()) == 2, 'Please make sure only a single sample is being passed to this function'
        
        responses = []
        for orientation in orientations:
            aspects = orientation[:,0]

            aspect_indices = torch.tensor((self.internal_num_viewing_geometries/np.pi)*aspects)

            floor_indices = aspect_indices.floor().long()
            frac = aspect_indices - floor_indices.float()
            ceil_indices = floor_indices + 1

            floor_indices = torch.clamp(floor_indices, 0, data.shape[0] - 1)
            ceil_indices = torch.clamp(ceil_indices, 0, data.shape[0] - 1)

            response = (1 - frac[..., None]) * data[floor_indices, :] + frac[...,None] * data[ceil_indices, :]

            responses.append(response)
        
        return responses
    
    def generateData(
            self,
            batch,
            orientations,
    ):
        range_profiles = self.forward(batch)
        del batch['data']

        
        response_data = []
        for i, data in enumerate(range_profiles):
            response_data.append(self.getResponse(data, orientations))

        #todo: write function to convert this info into form that can be saved into datasrt (i.e. xarray)
        return response_data, batch


