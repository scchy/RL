# python3
# Create Date: 2025-05-19
# Author: Scc_hy
# Func: A3C: Asynchronous Advantage Actor-Critic
# paper: [2016] Asynchronous Methods for Deep Reinforcement Learning 
#        https://arxiv.org/abs/1602.01783
# reference: hogwild: https://github.com/pytorch/examples/blob/main/mnist_hogwild/train.py
# ============================================================================================================================
import typing as typ
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from ._base_net import PPOValueNet as valueNet
from ._base_net import SACPolicyNet as policyNet


class A3C:
    """
    multiple actors leaners  & multiple env with different exploration
    Hogwild! style 
    """
    def __init__(
        self,
        state_dim: int,
        actor_hidden_layers_dim: typ.List[int],
        critic_hidden_layers_dim: typ.List[int],
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        A3C_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, action_bound=A3C_kwargs.get('action_bound', 1.0)).to(device)
        self.critic = valueNet(state_dim, critic_hidden_layers_dim, act_type=A3C_kwargs.get('act_type', 'relu')).to(device)
        














