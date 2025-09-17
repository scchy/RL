# python3
# Create Date: 2025-09-11
# Author: Scc_hy
# Fun: RND 
# reference: https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw5/cs285/agents/rnd_agent.py
# ============================================================================================


import torch
from torch import nn
import numpy as np
from typing import Callable, List, Tuple
from .dqn_agent import DQNAgent
from utils.utools import (
    from_numpy, to_numpy, device
)


def init_weight(model):
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()
    

class RNDAgent(DQNAgent):
    """
    用“一个随机初始化且永远固定的神经网络”作为神秘目标，再训练“另一个可学习的网络”去模仿它的输出；

    两网络在从未见过的状态上必然预测误差大，就把这个误差当成内在探索奖励——误差越大越“新鲜”，
    从而驱动策略主动去这些“陌生”地区收集数据。 
    """
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        make_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        make_rnd_network_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        make_target_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        rnd_weight: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.rnd_weight = rnd_weight

        self.rnd_net = make_rnd_network(observation_shape)
        self.rnd_target_net = make_target_rnd_network(observation_shape)

        self.rnd_target_net.apply(init_weight) # 固定随机网络
        self.rnd_optimizer = make_rnd_network_optimizer(
            self.rnd_net.parameters()
        )

    def update_rnd(self, obs: torch.Tensor):
        """
        Update the RND network using the observations.
        """
        loss = (self.rnd_net(obs) - self.rnd_target_net(obs).detach()).pow(2).mean()

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            # TODO(student): Compute RND bonus for batch and modify rewards
            rnd_error = (self.rnd_net(observations) - self.rnd_target_net(observations)).mean(dim=-1)
            rnd_error = (rnd_error - rnd_error.mean(dim=0)) / (rnd_error.std(dim=0) + 1e-8)  # batch normalize
            assert rnd_error.shape == rewards.shape
            rewards = rewards + self.rnd_weight * rnd_error

        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the RND network.
        rnd_loss = self.update_rnd(observations)
        metrics["rnd_loss"] = rnd_loss

        return metrics

    def num_aux_plots(self):
        return 1

    def plot_aux(
        self,
        axes: List,
    ):
        """
        Plot the RND prediction error for the observations.
        """
        import matplotlib.pyplot as plt
        assert len(axes) == 1
        ax: plt.Axes = axes[0]

        with torch.no_grad():
            # Assume a state space of [0, 1] x [0, 1]
            x = torch.linspace(0, 1, 100)
            y = torch.linspace(0, 1, 100)
            xx, yy = torch.meshgrid(x, y)

            inputs = from_numpy(np.stack([xx.flatten(), yy.flatten()], axis=1))
            targets = self.rnd_target_net(inputs)
            predictions = self.rnd_net(inputs)

            errors = torch.norm(predictions - targets, dim=-1)
            errors = torch.reshape(errors, xx.shape)

            # Log scale, aligned with normal axes
            from matplotlib import cm
            ax.imshow(to_numpy(errors).T, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
            plt.colorbar(ax.images[0], ax=ax)



