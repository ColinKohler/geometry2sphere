import math
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from typing import Tuple
from functools import partial

from e3nn import o3


from .gaussian_rbf import GaussianRadialBasisLayer, GaussianSmearing
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransformerBlock, 
)
from .input_block import EdgeDegreeEmbedding


# Statistics of IS2RE 100K 
_AVG_DEGREE = 3

def discretize(
    t: torch.Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> torch.Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

class Equiformerv2(nn.Module):
     
    def __init__(
        self,
        max_radius=5.0,

        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None, 

        edge_channels=128,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal'
    ):

        super().__init__()
        self.max_radius = max_radius
        self.cutoff = max_radius

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.edge_channels = edge_channels

        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cuda' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(0.0, self.cutoff, 600,2.0)
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(128, self.sphere_channels_all)

        self.discretize_coords = partial(discretize, num_discrete = 128, continuous_range = (-3.262670, 3.295396))
        #self.coor_embed = nn.Embedding(128, 64) 


        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.edge_channels_list,
            rescale_factor=_AVG_DEGREE
        )


        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransformerBlock(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        # self.final_ffn = FeedForwardNetwork(
        #     self.sphere_channels,
        #     self.ffn_hidden_channels, 
        #     1,
        #     self.lmax_list,
        #     self.mmax_list,
        #     self.SO3_grid,  
        #     self.ffn_activation,
        #     self.use_gate_act,
        #     self.use_grid_mlp,
        #     self.use_sep_s2_act
        # )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    
    def forward(self, data):
        self.batch_size = len(data.batch)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        num_nodes = int(data.batch.shape[0])

        edge_index = data.edge_index
        edge_distance_vec = data.edge_vec

        edge_distance = edge_distance_vec.norm(dim=1)

        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = SO3_Embedding(
            num_nodes,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        pos_y = x.clone()

        pos_z = x.clone()

        discretize_pos = self.discretize_coords(data['pos'])
        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(discretize_pos[:,0])
                pos_y.embedding[:, offset_res, :] = self.sphere_embedding(discretize_pos[:,1])
                pos_z.embedding[:, offset_res, :] = self.sphere_embedding(discretize_pos[:,2])
            else:
                x.embedding[:, :, offset_res, :] = self.sphere_embedding(discretize_pos)[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        
        edge_distance = self.distance_expansion(edge_distance)

        edge_degree = self.edge_degree_embedding(
            edge_distance,
            edge_index,
            num_nodes)
        
        x.embedding = x.embedding + pos_y.embedding + pos_z.embedding + edge_degree.embedding

        del pos_y, pos_z
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,
                edge_distance,
                edge_index,
                batch=data.batch
            )

        x.embedding = self.norm(x.embedding)

        output = x.embedding.view(x.embedding.shape[0], -1)

        # node_features = self.final_ffn(x)
        # node_features = node_features.embedding.narrow(1,0,1)
        # print(node_features.shape)
        # output = torch.zeros(num_nodes, device=node_features.device, dtype=node_features.dtype)
        # output.index_add_(0, data.batch, node_features.view(-1))

        return torch_geometric.utils.scatter(output, data.batch, dim=0, reduce='mean')

    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)