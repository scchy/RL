# python3 
# Create Date: 2025-07-16
# Func: Defines a pytorch policy as the agent's actor
# ===============================================================


import abc
import itertools
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch import distributions
from .base_policy import BasePolicy
from .utils import from_numpy, to_numpy


class ActorNN(nn.Module):
    """ 
    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions
    """
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size
    ):
        super(ActorNN, self).__init__()
        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.features = nn.ModuleList()
        in_size = self.ob_dim
        for _ in range(self.n_layers):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(in_size, self.size),
                'linear_action': nn.Tanh()
            }))
            in_size = self.size
        
        self.mean_head = nn.Linear(in_size, self.ac_dim)
        self.logstd = nn.Parameter(torch.zeros(1, ac_dim))
    
    def get_mean(self, x):
        for layer in self.features:
            x_lin = layer['linear'](x)
            x = layer['linear_action'](x_lin.clip(-200.0, 200.0))
        return self.mean_head(x)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        m_ = self.get_mean(observation)
        log_std = self.logstd.expand_as(m_) # self.log_std
        std_ = torch.exp(log_std)
        action_dist = distributions.Normal(m_, std_)
        return action_dist


class MLPPolicySL(BasePolicy, metaclass=abc.ABCMeta):
    """ 
    Defines an MLP for supervised learning which maps observations to continuous
    actions.
    """
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        device='cpu',
        **kwargs
    ):
        super(MLPPolicySL, self).__init__(**kwargs)
        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.device = kwargs.get('device', 'cpu')
        self.actor = ActorNN(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            n_layers=n_layers,
            size=size
        ).to(self.device)
        if training:
            self.actor.train()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.actor.state_dict(), filepath)
    
    @torch.no_grad()
    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        
        observation = from_numpy(observation.astype(np.float32)).to(self.device)
        action_dist = self.actor(observation)
        action = action_dist.sample()
        return to_numpy(action)

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        observations = torch.tensor(observations).float().to(self.device)
        actions = torch.tensor(actions).float().to(self.device)
        pred_dist = self.actor(observations)
        pred_a = pred_dist.rsample()

        # imitation loss
        loss = torch.mean(0.5 * (actions - pred_a) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': to_numpy(loss),
            'log_prob': to_numpy(pred_dist.log_prob(pred_a).mean())
        }

