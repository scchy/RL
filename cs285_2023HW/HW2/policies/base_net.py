import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from typing import Any
import numpy as np
import torch
from torch import distributions
from gymnasium.wrappers.normalize import RunningMeanStd
from .utils import from_numpy, to_numpy


def orthogonal_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


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
        self.register_parameter(name="obs_mean", param=nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False))
        self.register_parameter(name="obs_var", param=nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False))
        self.__init()

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

    def __init(self):
        for layer in self.features:
            orthogonal_init(layer['linear'], gain=np.sqrt(2))
            
        orthogonal_init(self.mean_head, gain=0.01)


class MLPPolicy:
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        device: str='cuda',
        norm_obs: bool=False,
        max_grad_norm: float=None
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm
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
        self.norm_obs = norm_obs
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.obs_rms = RunningMeanStd(shape=(ob_dim,))

    def normalize(self, obs, update_flag=True):
        if not self.norm_obs:
            return obs
        if update_flag:
            self.obs_rms.update(obs if isinstance(obs, np.ndarray) else obs.detach().cpu().numpy())
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-6)

    @torch.no_grad()
    def get_action(self, obs):
        obs = self.normalize(obs, False)
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
        if self.norm_obs:
            self.actor.obs_mean.data = torch.tensor(self.obs_rms.mean).to(self.device)
            self.actor.obs_var.data = torch.tensor(self.obs_rms.var).to(self.device)

        torch.save(self.actor.state_dict(), filepath)

    def load(self, filepath):
        actor_d = torch.load(filepath, map_location='cpu')
        self.actor.load_state_dict(actor_d)
        self.obs_rms.mean = actor_d['obs_mean'].detach().cpu().numpy()
        self.obs_rms.var = actor_d['obs_var'].detach().cpu().numpy()
        self.actor.to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

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
        obs = self.normalize(obs, True)
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
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) 
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
        self.__init()

    def forward(self, x: torch.FloatTensor) -> Any:
        for layer in self.features:
            x_lin = layer['linear'](x)
            x = layer['linear_action'](x_lin.clip(-200.0, 200.0))
        return self.v_head(x)

    def __init(self):
        for layer in self.features:
            orthogonal_init(layer['linear'], gain=np.sqrt(2))
            
        orthogonal_init(self.v_head, gain=1)


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        device: str='cuda',
        max_grad_norm: float=None
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm
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

        pred = self.network(obs).view(-1)
        # TODO: update the critic using the observations and q_values
        loss = 0.5 * (q_values - pred).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm) 
        self.optimizer.step()
        return {
            "Baseline Loss": to_numpy(loss),
            "vMax": to_numpy(pred.max()),
            "qMax": to_numpy(q_values.max()),
            "vMin": to_numpy(pred.min()),
            "qMin": to_numpy(q_values.min())
        }

