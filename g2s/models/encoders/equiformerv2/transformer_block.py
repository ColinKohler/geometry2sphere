from git import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
import copy

from .activation import (
    ScaledSiLU, 
    ScaledSwiGLU, 
    SwiGLU, 
    ScaledSmoothLeakyReLU, 
    SmoothLeakyReLU, 
    GateActivation,
    SeparableS2Activation, 
    S2Activation
)
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    get_normalization_layer
)
from .so2_ops import (
    SO2_Convolution, 
    SO2_Linear
)
from .so3 import (
    SO3_Embedding, 
    SO3_Linear, 
    SO3_LinearV2
)
from .radial_function import RadialFunction
from .drop import (
    GraphDropPath, 
    EquivariantDropoutArraySphericalHarmonics
)

class SO2EquivariantGraphAttention(torch.nn.Module):

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads, 
        attn_alpha_channels,
        attn_value_channels, 
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation, 
        mappingReduced, 
        SO3_grid, 
        edge_channels_list,
        use_m_share_rad=False,
        activation='scaled_silu', 
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        use_gate_act=False, 
        use_sep_s2_act=True,
        alpha_drop=0.0,
    ):
        
        
        super(SO2EquivariantGraphAttention, self).__init__()
        
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        self.use_s2_act_attn    = use_s2_act_attn
        self.use_attn_renorm    = use_attn_renorm
        self.use_gate_act       = use_gate_act
        self.use_sep_s2_act     = use_sep_s2_act

        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_m_share_rad = use_m_share_rad
        
        assert not self.use_s2_act_attn 

        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = extra_m0_output_channels + max(self.lmax_list) * self.hidden_channels
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad 
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad 
                else None
            ), 
            extra_m0_output_channels=extra_m0_output_channels # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
            #torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
    
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list), 
                mmax=max(self.mmax_list), 
                num_channels=self.hidden_channels
            )
        else:   
            if self.use_sep_s2_act:     
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), 
                    mmax=max(self.mmax_list)
                )
            else:                       
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), 
                    mmax=max(self.mmax_list)
                )
        
        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None, 
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn 
                else None
            ) # for attention weights
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels, self.output_channels, lmax=self.lmax_list[0])

    
    def forward(
        self,
        x, 
        edge_distance, 
        edge_index
    ):
        x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])
        
        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(), 
            x_target.num_channels * 2, 
            device=x_target.device, 
            dtype=x_target.dtype
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (max(self.lmax_list) + 1), 2 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index) # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)
        
        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:   
            # Gate activation
            x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
            x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
                x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights
                x_message.embedding = self.s2_act(x_0_gating, x_message.embedding, self.SO3_grid)
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            ##x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)
                
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)
        
        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha) # type: ignore
            x_0_alpha = self.alpha_act(x_0_alpha)  # type: ignore
            alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1]) # type: ignore
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        
        attn = x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1], len(x.embedding))

        # Project
        out_embedding = self.proj(x_message)

        return out_embedding
    

class FeedForwardNetwork(torch.nn.Module):
    def __init__(
        self,
        sphere_channels,
        hidden_channels, 
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,  
        activation='scaled_silu', 
        use_gate_act=False, 
        use_grid_mlp=False, 
        use_sep_s2_act=True
    ):
        super(FeedForwardNetwork, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)
        
        self.so3_linear_1 = SO3_LinearV2(self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax)
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(self.sphere_channels_all, self.hidden_channels, bias=True), 
                    nn.SiLU(), 
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False), 
                nn.SiLU(), 
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(), 
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(self.sphere_channels_all, self.max_lmax * self.hidden_channels)
                self.gate_act = GateActivation(self.max_lmax, self.max_lmax, self.hidden_channels)
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(self.sphere_channels_all, self.hidden_channels)
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.max_lmax)
        
    
    def forward(self, input_embedding):

        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(input_embedding.embedding.narrow(1, 0, 1)) # type: ignore 
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)
        
        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(self.SO3_grid, lmax=self.max_lmax)
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(input_embedding_grid, self.SO3_grid, lmax=self.max_lmax)

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat((gating_scalars, input_embedding.embedding.narrow(1, 1, input_embedding.embedding.shape[1] - 1)), dim=1) # type: ignore
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(gating_scalars, input_embedding.embedding)
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(gating_scalars, input_embedding.embedding, self.SO3_grid)
                else:
                    input_embedding.embedding = self.s2_act(input_embedding.embedding, self.SO3_grid)

        input_embedding = self.so3_linear_2(input_embedding)

        return input_embedding
    

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels, 
        attn_value_channels,
        ffn_hidden_channels,
        output_channels, 

        lmax_list,
        mmax_list,
        
        SO3_rotation,
        mappingReduced,
        SO3_grid,

        edge_channels_list,
        use_m_share_rad=False,

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False, 
        use_grid_mlp=False,
        use_sep_s2_act=True,

        norm_type='rms_norm_sh',

        alpha_drop=0.0, 
        drop_path_rate=0.0, 
        proj_drop=0.0
    ):
        super(TransformerBlock, self).__init__()

        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads, 
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels, 
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation, 
            mappingReduced=mappingReduced, 
            SO3_grid=SO3_grid, 
            edge_channels_list=edge_channels_list,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation, 
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )


        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.proj_drop = EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)
        
        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels, 
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,  
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(sphere_channels, output_channels, lmax=max_lmax)
        else:
            self.ffn_shortcut = None


    def forward(
            self,
            x,
            edge_distance,
            edge_index,
            batch
    ):
        output_embedding = x

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(output_embedding, edge_distance, edge_index)
        
        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0, 
                output_embedding.lmax_list.copy(), 
                self.ffn_shortcut.in_features, 
                device=output_embedding.device, 
                dtype=output_embedding.dtype
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(output_embedding.lmax_list.copy(), output_embedding.lmax_list.copy())
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding