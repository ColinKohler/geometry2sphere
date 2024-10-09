""" object_2_sphere.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3

from torch_harmonics.spherical_harmonics_old import SphericalHarmonics
from torch_harmonics.spherical_harmonics_mlp import SphericalHarmonics as SHMLP

from g2s.models import e3nn_utils
from g2s.models.modules import MLP, S2MLP, SphericalCNN
from g2s.models.encoders.equiformerv2.equiformerv2 import Equiformerv2


class Mesh2Radar(nn.Module):
    """Mesh2Sphere model. Converts betweeen 3D object meshes to spherical signals.

    Args:
        latent_lmax:
        ouput_lmax:
        latent_feat_dim:
        max_radius:
        num_out_spheres:
    """

    def __init__(
        self,
        latent_lmax: int,
        output_lmax: int,
        latent_feat_dim: int,
        max_radius: float,
        num_out_spheres: int = 1,
        use_mlp: bool = True,
        num_layers_equivformer: int = 4,
        num_heads_equivformer: int = 4,
        num_theta: int = 1,
        num_phi: int = 1,
    ):
        super().__init__()

        self.latent_lmax = latent_lmax
        self.output_lmax = output_lmax
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres

        self.irreps_enc_out = o3.Irreps(
            [
                (latent_feat_dim, (l, p))
                for l in range((latent_lmax) + 1)
                for p in [-1, 1]
            ]
        )
        self.encoder = Equiformerv2(
            num_layers=num_layers_equivformer,
            num_heads=num_heads_equivformer,
            ffn_hidden_channels=latent_feat_dim,
            lmax_list=[latent_lmax],
            max_radius=max_radius,
        )

        self.spherical_cnn = SphericalCNN(
            [
                latent_lmax,
                5,
                10,
                20,
                output_lmax,
                output_lmax,
            ],
            [
                latent_feat_dim,
                64,
                32,
                16,
                num_out_spheres,
                num_out_spheres,
            ],
        )

        self.sh = SphericalHarmonics(
            L=output_lmax, grid_type="linear", num_theta=num_theta, num_phi=num_phi
        )
        # self.sh = SHMLP(
        #    L=output_lmax, grid_type="linear", num_theta=num_theta, num_phi=num_phi
        # )

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP([num_out_spheres, num_out_spheres])

    def forward(self, x):
        B = x.batch_size

        z = self.encoder(x)
        w = self.spherical_cnn(z.view(B, 1, -1))
        out = self.sh(w).permute(0, 1, 3, 2)
        if self.use_mlp:
            out = self.mlp(out.permute(0, 2, 3, 1))
            out = out.permute(0, 3, 1, 2)

        return out, w


class Mesh2Drag(nn.Module):
    """Mesh2Sphere model. Converts betweeen 3D object meshes to spherical signals.

    Args:
        latent_lmax:
        ouput_lmax:
        latent_feat_dim:
        max_radius:
        num_out_spheres:
    """

    def __init__(
        self,
        latent_lmax: int,
        output_lmax: int,
        latent_feat_dim: int,
        max_radius: float,
        num_out_spheres: int = 1,
        num_layers_equivformer: int = 4,
        num_heads_equivformer: int = 4,
        num_theta: int = 1,
        num_phi: int = 1,
    ):
        super().__init__()

        self.latent_lmax = latent_lmax
        self.output_lmax = output_lmax
        self.num_theta = num_theta
        self.num_phi = num_phi
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres

        self.irreps_enc_out = o3.Irreps(
            [(latent_feat_dim, (l, 1)) for l in range((latent_lmax) + 1)]
        )
        self.irreps_out = o3.Irreps([(1, (l, 1)) for l in range((latent_lmax) + 1)])
        self.encoder = Equiformerv2(
            num_layers=num_layers_equivformer,
            num_heads=num_heads_equivformer,
            ffn_hidden_channels=latent_feat_dim,
            lmax_list=[latent_lmax],
            max_radius=max_radius,
        )

        self.lin = o3.Linear(
            (o3.Irreps("3x0e") + self.irreps_enc_out).simplify(), self.irreps_enc_out
        )

        self.spherical_cnn = SphericalCNN(
            [
                latent_lmax,
                output_lmax,
                output_lmax,
                output_lmax,
                output_lmax,
            ],
            [
                latent_feat_dim,
                64,
                32,
                16,
                num_out_spheres,
            ],
        )

        self.sh = SphericalHarmonics(
            L=output_lmax, grid_type="linear", num_theta=num_theta, num_phi=num_phi
        )

    def forward(self, x):
        x, flight_cond, coords = x
        B = x.batch_size

        z = self.encoder(x)
        z = torch.concat([flight_cond, z.view(B, -1)], dim=-1)
        z = self.lin(z)
        w = self.spherical_cnn(z.view(B, 1, -1))
        out = self.sh(w.view(B, self.num_out_spheres, -1), coords)

        return out, w


if __name__ == "__main__":
    m2s = Mesh2Radar(2, 10, 32, 1.5)
    print(type(m2s))
