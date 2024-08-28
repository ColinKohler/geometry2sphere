import torch
import torch.nn as nn
import torch.nn.functional as F


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
