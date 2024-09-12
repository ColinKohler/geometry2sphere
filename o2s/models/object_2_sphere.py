""" object_2_sphere.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3

from torch_harmonics.spherical_harmonics_old import SphericalHarmonics

from o2s.models import e3nn_utils
from o2s.models.modules import MLP, SphericalCNN
from o2s.models.encoders.equiformerv2.equiformerv2 import Equiformerv2


class Mesh2Sphere(nn.Module):
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
        mlp_hidden_dim=[61 * 2 * 21, 1024, 61 * 2 * 21],
        num_layers_equivformer: int = 4,
        num_heads_equivformer: int = 4,
        num_lat: int = 21,
        num_lon: int = 61
    ):
        super().__init__()

        self.latent_lmax = latent_lmax
        self.output_lmax = output_lmax
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres

        self.irreps_latent = e3nn_utils.so3_irreps(latent_lmax)
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
        # self.irreps_enc_out = e3nn_utils.s2_irreps(z_lmax)

        self.spherical_cnn = SphericalCNN(
            [latent_lmax, output_lmax // 4, output_lmax // 2, output_lmax],
            # [latent_lmax, output_lmax, output_lmax, output_lmax],
            [latent_feat_dim, 64, 32, num_out_spheres],
        )
        self.lin = o3.Linear(
            e3nn_utils.s2_irreps(output_lmax),
            e3nn_utils.s2_irreps(output_lmax),
            f_in=num_out_spheres,
            f_out=num_out_spheres,
            biases=True,
        )
        self.sh = SphericalHarmonics(
            output_lmax, output_lmax + 1, num_lat=num_lat, num_lon=num_lon
        )

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(mlp_hidden_dim)

    def forward(self, x):
        B = x.batch_size

        z = self.encoder(x)
        w = self.spherical_cnn(z.view(B, 1, -1))
        w = self.lin(w)
        out = self.sh(w.squeeze())
        if self.use_mlp:
            out = self.mlp(out.float().view(B, -1)).view(B, 61 * 2, 21)

        return out, w


if __name__ == "__main__":
    m2s = Mesh2Sphere(2, 10, 32, 1.5)
    print(type(m2s))
