
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
import torch
from .dqn_basic_config import basic_dqn_config
from utils.utools import (
    from_numpy, to_numpy, build_mlp
)
from base_net.mlp_policy import MLPPolicy


def awac_config(
    total_steps: int = 50000,
    discount: float = 0.95,
    temperature: float = 1.0,
    actor_hidden_size: int = 128,
    actor_num_layers: int = 2,
    actor_learning_rate: float = 3e-4,
    **kwargs,
):
    make_actor = lambda obs_shape, num_actions: MLPPolicy(
        num_actions,
        obs_shape[0],
        discrete=True,
        n_layers=actor_num_layers,
        layer_size=actor_hidden_size,
    )
    make_actor_optimizer = lambda params: torch.optim.Adam(params, lr=actor_learning_rate)

    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["log_name"] = "{env_name}_awac{temperature}".format(
        env_name=config["env_name"], temperature=temperature
    )
    config["agent"] = "awac"

    config["agent_kwargs"]["temperature"] = temperature
    config["agent_kwargs"]["make_actor"] = make_actor
    config["agent_kwargs"]["make_actor_optimizer"] = make_actor_optimizer

    return config

