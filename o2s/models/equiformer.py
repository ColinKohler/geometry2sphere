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
        num_out_spheres: int = 1,
    ):
        super().__init__()

        self.lmax = lmax
        self.latent_feat_dim = latent_feat_dim
        self.num_out_spheres = num_out_spheres

        self.irreps_latent = e3nn_utils.so3_irreps(lmax)
        self.irreps_enc_out = o3.Irreps(
            [(latent_feat_dim, (l, p)) for l in range((lmax) + 1) for p in [-1, 1]]
        )
        self.encoder = Equiformerv2(
            num_layers=2,
            num_heads=4,
            ffn_hidden_channels=latent_feat_dim,
            lmax_list=[lmax],
            max_radius=max_radius,
        )
        self.spherical_cnn = SphericalCNN(
            [lmax, lmax, lmax, lmax],
            [latent_feat_dim, 64, 32, num_out_spheres],
        )
        self.lin = o3.Linear(
            e3nn_utils.s2_irreps(lmax),
            o3.Irreps(str(61 * 2 * 21) + "e"),
            f_in=latent_feat_dim,
            f_out=num_out_spheres,
            biases=True,
        )

    def forward(self, x):
        B = x.batch_size

        z = self.encoder(x)
        out = self.spherical_cnn(z.view(B, 1, -1))
        out = self.lin(w)

        return out


if __name__ == "__main__":
    equiformer = EquiformerDecoder(2, 32, 1.5)
    print(type(equiformer))
