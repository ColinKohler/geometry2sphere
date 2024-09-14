""" equifromer.py """

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3

from o2s.models import e3nn_utils
from o2s.models.modules import MLP, SphericalCNN
from o2s.models.encoders.equiformerv2.equiformerv2 import Equiformerv2


class EquiformerDecoder(nn.Module):
    """Equiformer baseline model.

    Args:
        lmax:
        latent_feat_dim:
        max_radius:
        num_out_spheres:
    """

    def __init__(
        self,
        lmax: int,
        latent_feat_dim: int,
        max_radius: float,
        num_layers: int = 4,
        num_heads: int = 4,
        num_out_spheres: int = 1,
        num_theta: int = 61,
        num_phi: int = 21,
    ):
        super().__init__()

        self.lmax = lmax
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres
        self.num_theta = num_theta
        self.num_phi = num_phi

        self.irreps_latent = e3nn_utils.so3_irreps(lmax)
        self.irreps_enc_out = o3.Irreps(
            [(latent_feat_dim, (l, p)) for l in range((lmax) + 1) for p in [-1, 1]]
        )
        self.encoder = Equiformerv2(
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_channels=latent_feat_dim,
            lmax_list=[lmax],
            max_radius=max_radius,
        )
        self.spherical_cnn = SphericalCNN(
            [lmax, lmax, lmax, lmax],
            [latent_feat_dim, 64, 32, 16],
        )
        self.lin = o3.Linear(
            e3nn_utils.s2_irreps(lmax),
            o3.Irreps(str(num_theta * num_phi) + "x0e"),
            f_in=16,
            f_out=num_out_spheres,
            biases=True,
        )

    def forward(self, x):
        B = x.batch_size

        z = self.encoder(x)
        out = self.spherical_cnn(z.view(B, 1, -1))
        out = self.lin(out)
        out = out.view(B, self.num_out_spheres, self.num_theta, self.num_phi)

        return out, None


if __name__ == "__main__":
    equiformer = EquiformerDecoder(2, 32, 1.5)
    print(type(equiformer))
