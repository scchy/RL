import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from typing import Any
import numpy as np
import torch
from torch import distributions
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
        size,
        discrete=False
    ):
        super(ActorNN, self).__init__()
        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.discrete = discrete
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
        if self.discrete:
            return distributions.Categorical(logits=m_) 
        log_std = self.logstd.expand_as(m_) # self.log_std
        std_ = torch.exp(log_std)
        action_dist = distributions.Normal(m_, std_)
        return action_dist


class MLPPolicy:
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        device: str='cuda'
    ):
        super().__init__()
        self.device = device
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.actor = ActorNN(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            n_layers=n_layers,
            size=layer_size,
            discrete=discrete
        ).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        
        observation = from_numpy(observation.astype(np.float32)).to(self.device)
        actor_res = self.actor(observation)
        if self.discrete:
            action = actor_res.sample()
            return to_numpy(action)

        action = actor_res.sample()
        return to_numpy(action)
    
    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.actor.state_dict(), filepath)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = from_numpy(obs).to(self.device)
        actions = from_numpy(actions).to(self.device)
        advantages = from_numpy(advantages).to(self.device)

        pred_dist = self.actor(obs)
        if self.discrete:
            # TODO: discrete log proba 
            log_proba = pred_dist.log_prob(actions)
        else:
            # pred_a = pred_dist.rsample()
            log_proba = pred_dist.log_prob(actions).sum(1)

        # TODO: implement the policy gradient actor update.
        # print(f'{advantages.shape=} {log_proba.shape=}')
        loss = -torch.mean(log_proba * advantages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": to_numpy(loss),
        }

    def train(self):
        self.actor.train()
    
    def eval(self):
        self.actor.eval()


class criticNN(nn.Module):
    def __init__(
        self,
        ob_dim,
        n_layers,
        size
    ):
        super(criticNN, self).__init__()
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.features = nn.ModuleList()
        in_size = ob_dim
        for _ in range(self.n_layers):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(in_size, self.size),
                'linear_action': nn.Tanh()
            }))
            in_size = self.size
        self.v_head = nn.Linear(in_size, 1)

    def forward(self, x: torch.FloatTensor) -> Any:
        for layer in self.features:
            x_lin = layer['linear'](x)
            x = layer['linear_action'](x_lin.clip(-200.0, 200.0))
        return self.v_head(x)


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        device: str='cuda'
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ob_dim = ob_dim
        self.device = device
        self.learning_rate = learning_rate
        self.network = criticNN(
            ob_dim=ob_dim,
            n_layers=n_layers,
            size=layer_size,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )
        self.train()
    
    def eval(self):
        self.network.eval()
    
    def train(self):
        self.network.train()

    def forward(self, obs):
        return self.network(obs)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = from_numpy(obs).to(self.device)
        q_values = from_numpy(q_values).to(self.device)

        pred = self.network(obs)
        # TODO: update the critic using the observations and q_values
        loss = 0.5 * (q_values - pred).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Baseline Loss": to_numpy(loss),
        }

