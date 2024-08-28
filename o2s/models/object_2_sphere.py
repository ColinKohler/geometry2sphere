from sympy import use
import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import nn as enn
from e3nn import o3
from e3nn.nn import SO3Activation

from o2s.models import e3nn_utils
from o2s.models.encoders.equiformerv2 import Equiformerv2


class MLP(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        layers = []
        for hid_dim in hidden_dim:
            layers.append(nn.Linear(in_features=input_dim, out_features=hid_dim))
            layers.append(nn.ReLU())
            input_dim = hid_dim

        layers.append(nn.Linear(in_features=input_dim, out_features=self.input_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(
        self,
        lmax_in,
        lmax_out,
        f_in,
        f_out,
        lmax_list=[10, 15, 20],
        invariant_out=False,
    ):
        super().__init__()
        self.invariant_out = invariant_out

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
        if self.invariant_out:
            self.act3 = SO3Activation(lmax_in, 0, torch.relu, resolution=12)
            self.lin = o3.Linear(256, f_out)
        else:
            self.act3 = SO3Activation(
                lmax_list[2], lmax_list[2], torch.relu, resolution=12
            )
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
        z_lmax,
        max_radius,
        out_dim,
        f,
        encoder,
        lmax_out=None,
        use_mlp=True,
        mlp_hidden_dim=[256, 512, 512],
        invariant_out=False,
    ):
        super().__init__()

        #    z_lmax = 4
        self.lmax = z_lmax
        self.out_dim = out_dim
        self.invariant_out = invariant_out
        self.f = f
        self.lmax_out = lmax_out
        if self.lmax_out == None:
            self.lmax_out = z_lmax

        self.irreps_in = o3.Irreps("1x0e")
        self.irreps_latent = e3nn_utils.so3_irreps(z_lmax)
        self.irreps_enc_out = o3.Irreps(
            # [(f, (l, p)) for l in range((z_lmax // 2) + 1) for p in [-1,1]]
            [(f[0], (l, p)) for l in range((z_lmax) + 1) for p in [-1, 1]]
        )
        if encoder == "gnn":
            if self.invariant_out:
                self.irreps_node_attr = o3.Irreps("1x1e")
                self.encoder = AttrGNN(
                    irreps_node_input=self.irreps_in,
                    irreps_node_attr=self.irreps_node_attr,
                    irreps_node_output=self.irreps_enc_out,
                    max_radius=max_radius,
                    layers=2,
                    mul=f[0],
                    lmax=[self.lmax, self.lmax, self.lmax],  # type: ignore
                )
            else:
                self.encoder = GNN(
                    irreps_node_input=self.irreps_in,
                    irreps_node_output=self.irreps_enc_out,
                    max_radius=max_radius,
                    layers=2,
                    mul=f[0],
                    # lmax=[self.lmax // 2, self.lmax // 2, self.lmax // 2],
                    lmax=[self.lmax, self.lmax, self.lmax],  # type: ignore
                )

        # elif isinstance(encoder, Equiformer):
        #   self.encoder = encoder(irreps_in=self.irreps_in, irreps_feature=self.irreps_enc_out, max_radius=max_radius, mean=0, std=1)

        else:
            self.encoder = encoder(lmax_list=[self.lmax], max_radius=max_radius)
            self.irreps_enc_out = e3nn_utils.s2_irreps(z_lmax)

        # else:
        #   print(type(encoder))
        #   raise ValueError

        linear_layers = []
        for i in range(len(f)):
            if i == 0:
                f_in = 128
                f_out = f[i]
                input_irrep = self.irreps_enc_out
            else:
                f_in = f[i - 1]
                f_out = f[i]
                input_irrep = self.irreps_latent
            linear_layers.append(
                o3.Linear(input_irrep, self.irreps_latent, f_in=f_in, f_out=f_out)
            )
            if i != len(f) - 1:
                linear_layers.append(
                    SO3Activation(z_lmax, z_lmax, torch.relu, resolution=12)
                )
        self.linear_layers = nn.Sequential(*linear_layers)
        self.decoder = Decoder(
            z_lmax, self.lmax_out, f[-1], out_dim, invariant_out=invariant_out
        )

        self.ls = [i for i in range(self.lmax_out + 1)]

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(hidden_dim=mlp_hidden_dim, input_dim=self.out_dim)

    def forward(self, x, return_latent=False):
        batch_size = x.batch.max() + 1
        enc_out = self.encoder(x)

        z = self.linear_layers(enc_out.view(batch_size, 1, -1))
        out = self.decoder(z)

        if return_latent:
            return out, z
        else:
            return out

    def getResponse(self, out, pose, get_seprate_responses=False):
        if self.invariant_out:
            return out
        else:

            sh = o3.spherical_harmonics(self.ls, pose, True)
            # sh_mlp = self.mlp(sh)

            sh = rearrange(sh, "b n l -> b l n")
            # sh_mlp = rearrange(sh_mlp, "b n l -> b l n")

            response = out @ sh
            # response_1 = out[:, self.out_dim : ] @ sh
            # response_2 = out[:, : self.out_dim] @ sh_mlp
            # response = response_1 + response_2
            # response = out @ sh

            response = rearrange(response, "b F n -> b n F").contiguous()
            if self.use_mlp:
                response = self.mlp(response)
            # if get_seprate_responses:
            # return response_1, response_2

            return response


if __name__ == "__main__":
    o2s = Object2Sphere(2, 1.8, 107, f=6, encoder="gnn")
    print(type(o2s))
