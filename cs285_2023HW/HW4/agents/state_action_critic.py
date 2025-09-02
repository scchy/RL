import torch
from torch import nn

from utils.utools import device, build_mlp


class StateActionCritic(nn.Module):
    def __init__(self, ob_dim, ac_dim, n_layers, size):
        super().__init__()
        self.net = build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(device)
    
    def forward(self, obs, acs):
        return self.net(torch.cat([obs, acs], dim=-1)).squeeze(-1)

