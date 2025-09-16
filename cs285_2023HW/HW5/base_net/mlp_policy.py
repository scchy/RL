# python3
# Create Date: 2025-09-16
# Author: Scc_hy
# Func: actor policy
# Refrence: https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw5/cs285/networks/mlp_policy.py
# ==================================================================================================


from typing import Optional

from torch import nn
import torch
from torch import distributions
from utils.utools import (
    from_numpy, to_numpy, build_mlp, device
)
from utils.distributions import make_tanh_transformed, make_multi_normal


class MLPPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()
        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std
        if discrete:
            self.logits_net = build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(device)
            else:
                self.net = build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(device)   

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=device)
                    )

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution

