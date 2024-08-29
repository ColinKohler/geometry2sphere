""" modules.py """

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3
from e3nn.nn import SO3Activation

from o2s.models import e3nn_utils


class MLP(nn.Module):
    """Simple MLP."""

    def __init__(
        self,
        hiddens: List[int],
        dropout: float = 0.0,
        act_out: bool = True,
    ):
        super().__init__()

        layers = list()
        for i, (h, h_) in enumerate(zip(hiddens, hiddens[1:])):
            layers.append(nn.Linear(h, h_))
            is_last_layer = i == len(hiddens) - 2
            if not is_last_layer or act_out:
                layers.append(nn.LeakyReLU(0.01, inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class S2Convolution(torch.nn.Module):
    """S2 Convolution"""

    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y",
            o3.spherical_harmonics_alpha_beta(
                range(lmax + 1), *kernel_grid, normalization="component"
            ),
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(
            e3nn_utils.s2_irreps(lmax),
            e3nn_utils.so3_irreps(lmax),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(nn.Module):
    """SO3 Convolution

    Args:
        f_in:
        f_out:
        lmax_in:
        kernel_grid:
        lmax_out:
    """

    def __init__(
        self, f_in: int, f_out: int, lmax_in: int, kernel_grid, lmax_out: int = None
    ):
        super().__init__()
        if lmax_out == None:
            lmax_out = lmax_in
        self.register_parameter(
            "w", nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_parameter(
            "D", nn.Parameter(e3nn_utils.flat_wigner(lmax_in, *kernel_grid))
        )  # [n_so3_pts, psi]
        self.lin = o3.Linear(
            e3nn_utils.so3_irreps(lmax_in),
            e3nn_utils.so3_irreps(lmax_out),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3ToS2Convolution(nn.Module):
    """SO3 to S2 Linear Layer"""

    def __init__(self, f_in: int, f_out: int, lmax_in: int, lmax_out: int, kernel_grid):
        super().__init__()
        s2_irreps = o3.Irreps(
            [(1, (l, 1)) for l in range(lmax_out + 1)]
            + [(1, (l, 1)) for l in range(lmax_out + 1)]
        )
        self.lin = o3.Linear(
            e3nn_utils.so3_irreps(lmax_in),
            e3nn_utils.s2_irreps(lmax_out),
            # s2_irreps,
            f_in=f_in,
            f_out=f_out,
        )

    def forward(self, x):
        return self.lin(x)


class SphericalCNN(nn.Module):
    """Spherical CNN.

    Args:
        lmax:
        feat:
    """

    def __init__(self, lmax: List[int], feat: List[int], N: int = 8):
        super().__init__()

        # Need the same number of lmax and feats for layers
        assert len(lmax) == len(feat)

        grid_s2 = e3nn_utils.s2_near_identity_grid()
        grid_so3 = e3nn_utils.so3_near_identity_grid()

        self.s2_conv = S2Convolution(feat[0], feat[0], lmax[0], grid_s2)
        self.s2_act = SO3Activation(lmax[0], lmax[0], torch.relu, resolution=N)

        layers: list = []
        for i, (f, f_) in enumerate(zip(feat[:-1], feat[1:-1])):
            layers.append(
                SO3Convolution(
                    f,
                    f_,
                    lmax_in=lmax[i],
                    lmax_out=lmax[i + 1],
                    kernel_grid=grid_so3,
                )
            )

            layers.append(
                SO3Activation(lmax[i + 1], lmax[i + 1], torch.relu, resolution=N)
            )
        self.so3_conv = nn.Sequential(*layers)

        self.lin = SO3ToS2Convolution(
            feat[-2],
            feat[-1],
            lmax_in=lmax[-2],
            lmax_out=lmax[-1],
            kernel_grid=grid_so3,
        )

    def forward(self, x):
        x = self.s2_conv(x)
        x = self.s2_act(x)
        x = self.so3_conv(x)
        x = self.lin(x)

        return x
