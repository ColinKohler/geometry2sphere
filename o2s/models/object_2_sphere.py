""" object_2_sphere.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3
from e3nn.nn import SO3Activation

from torch_harmonics.spherical_harmonics import SphericalHarmonics

from o2s.models import e3nn_utils
from o2s.models.modules import MLP
from o2s.models.encoders.equiformerv2.equiformerv2 import Equiformerv2


class SphericalCNN(nn.Module):
    def __init__(
        self,
        lmax_in,
        lmax_out,
        f_in,
        f_out,
        lmax_list=[10, 15, 20],
    ):
        super().__init__()

        grid_s2 = e3nn_utils.s2_near_identity_grid()
        grid_so3 = e3nn_utils.so3_near_identity_grid()

        self.so3_conv1 = e3nn_utils.SO3Convolution(
            f_in, 128, lmax_in=lmax_in, lmax_out=lmax_list[0], kernel_grid=grid_so3
        )
        self.act1 = SO3Activation(lmax_list[0], lmax_list[0], torch.relu, resolution=12)

        self.so3_conv2 = e3nn_utils.SO3Convolution(
            128, 256, lmax_in=lmax_list[0], lmax_out=lmax_list[1], kernel_grid=grid_so3
        )
        self.act2 = SO3Activation(lmax_list[1], lmax_list[1], torch.relu, resolution=12)

        self.so3_conv3 = e3nn_utils.SO3Convolution(
            256, 512, lmax_in=lmax_list[1], lmax_out=lmax_list[2], kernel_grid=grid_so3
        )

        # Output: Maps to 53 (rho_0, rho_1, rho_2, rho_3, ...) -> 53 S2 signals
        self.act3 = SO3Activation(lmax_list[2], lmax_list[2], torch.relu, resolution=12)
        self.lin = e3nn_utils.SO3ToS2Convolution(
            512, f_out, lmax_in=lmax_list[2], lmax_out=lmax_out, kernel_grid=grid_s2
        )

    def forward(self, x):
        x = self.so3_conv1(x)
        x = self.act1(x)

        x = self.so3_conv2(x)
        x = self.act2(x)

        x = self.so3_conv3(x)
        x = self.act3(x)

        x = self.lin(x)

        return x


class Object2Sphere(nn.Module):
    def __init__(
        self,
        latent_lmax,
        output_lmax,
        latent_feat_dim,
        max_radius,
        num_out_spheres=1,
        use_mlp=True,
        mlp_hidden_dim=[256, 512, 512],
    ):
        super().__init__()

        self.latent_lmax = latent_lmax
        self.output_lmax = output_lmax
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres

        self.irreps_in = o3.Irreps("1x0e")
        self.irreps_latent = e3nn_utils.so3_irreps(latent_lmax)
        self.irreps_enc_out = o3.Irreps(
            [
                (latent_feat_dim, (l, p))
                for l in range((latent_lmax) + 1)
                for p in [-1, 1]
            ]
        )
        self.encoder = Equiformerv2(lmax_list=[output_lmax], max_radius=max_radius)
        # self.irreps_enc_out = e3nn_utils.s2_irreps(z_lmax)

        self.spherical_cnn = SphericalCNN(
            latent_lmax, output_lmax, latent_feat_dim, num_out_spheres
        )
        self.sh = SphericalHarmonics(self.lmax_out, num_lat=0, num_lon=0)

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(hidden_dim=mlp_hidden_dim, input_dim=self.out_dim)

    def forward(self, x):
        batch_size = x.batch.max() + 1
        enc_out = self.encoder(x)

        z = self.linear_layers(enc_out.view(batch_size, 1, -1))
        w = self.spherical_cnn(z)
        out = self.sh(w)

        return out, w


if __name__ == "__main__":
    o2s = Object2Sphere(2, 10, 32, 1.5)
    print(type(o2s))
