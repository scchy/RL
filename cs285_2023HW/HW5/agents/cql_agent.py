# python3
# Create Date: 2025-09-15
# Author: Scc_hy
# Fun: CQL 
# reference:
# ============================================================================================


import torch
from torch import nn
import numpy as np
from typing import Callable, List, Tuple, Sequence
from .dqn_agent import DQNAgent
from utils.utools import (
    from_numpy, to_numpy, device
)


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature
    
    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )
        # TODO(student): modify the loss to implement CQL

        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        q_sa = self.critic(obs)
        loss = loss + self.cql_alpha * (
            torch.logsumexp(q_sa/self.cql_temperature) - q_sa.gather(1, action.long().view(-1, 1))
        )

        return loss, metrics, variables