

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
import torch
from .dqn_basic_config import basic_dqn_config
from utils.utools import (
    from_numpy, to_numpy, build_mlp
)
from .awac_config import awac_config


def iql_config(
    total_steps: int = 50000,
    discount: float = 0.95,
    temperature: float = 1.0,
    expectile: float = 0.9,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 3e-4,
    **kwargs,
):
    config = awac_config(
        total_steps=total_steps,
        discount=discount,
        temperature=temperature,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        **kwargs,
    )
    config["log_name"] = "{env_name}_iql{expectile}_temp{temperature}".format(
        env_name=config["env_name"], temperature=temperature, expectile=expectile
    )
    config["agent"] = "iql"

    config["agent_kwargs"]["expectile"] = expectile
    config["agent_kwargs"]["temperature"] = temperature

    config["agent_kwargs"]["make_value_critic"] = lambda obs_shape: build_mlp(
        input_size=np.prod(obs_shape),
        output_size=1,
        n_layers=num_layers,
        size=hidden_size,
    )
    config["agent_kwargs"][
        "make_value_critic_optimizer"
    ] = lambda params: torch.optim.Adam(params, lr=learning_rate)

    return config


