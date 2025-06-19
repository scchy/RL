# python3
# Author: Scc_hy
# Create Date: 2025-06-10
# Func: Soft Q-Learning
#       its structure resembles an actor-critic algorithm.
# reference: https://github.com/haarnoja/softqlearning/blob/master/softqlearning/algorithms/sql.py
# ============================================================================
import typing as typ
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Uniform 
from ._base_net import DDPGValueNet as valueNet
from ._base_net import SACPolicyNet as policyNet
from copy import deepcopy


class SQL:
    """
    [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
    
    \theta: Q_{soft} & target Q_{soft}
    \phi: \pi_{soft}
     
     
    """
    def __init__(
        self,
        state_dim: int,
        actor_hidden_layers_dim: typ.List[int],
        critic_hidden_layers_dim: typ.List[int],
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        SQL_kwargs: typ.Dict,
        device: torch.device  
    ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device
        self.tau = SQL_kwargs.get('tau', 0.05)
        self.alpha = SQL_kwargs.get('alpha', 0.25)

        self.q = valueNet(state_dim+action_dim, critic_hidden_layers_dim).to(device)
        self.tar_q = deepcopy(self.q)
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, action_bound=SQL_kwargs.get('action_bound', 1.0)).to(device)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action = self.actor(state)[0]
        return action.cpu().detach().numpy()[0]
    
    def update(self, samples):
        state, action, reward, next_state, done = zip(*samples)

        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.tensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)
        
        # ------------------------------------------------
        # QLoss
        # next state with uniform samples
        new_next_act, _ = self.actor(next_state)
        next_v = next_q = self.tar_q(next_state, new_next_act)
        # Eqution 10: \alpha log sum_ [exp(1/alpha * Q)/q_a(a)] 
        # next_v = self.alpha * torch.logsumexp(1/self.alpha * next_q, dim=1) - torch.log(self.value_n_particles)
        # + np.log(2 ** action_dim) # why
        tar_q = reward + self.gamma * next_v * (1 - done)
        q_v = self.q(state, action) 
        # q loss 
        self.q_opt.zero_grad()
        q_loss = (0.5 * (tar_q - q_v) ** 2).mean()
        q_loss.backward()
        self.q_opt.step()
        # ------------------------------------------------
        # Actor loss: svg update
        self._tanh_guassion_update(state) #self._SVG_update()
        # self.soft_update(self.actor, self.tar_actor)
        self.soft_update(self.q, self.tar_q)

    def _tanh_guassion_update(self, state):
        new_act, log_prob = self.actor(state)
        q_v = self.tar_q(state, new_act)
        a_loss = torch.mean(self.alpha * log_prob - q_v)

        self.actor_opt.zero_grad()
        # a_loss = (log_p.exp() * (log_p - log_q)).mean()
        a_loss.backward()
        self.actor_opt.step()

    def _SVG_update(self):
        a = self.actor(state)
        actions_fixed = self.actor(states, self.kernel_K_fixed)
        actions_updated = self.actor(states, self.kernel_K_updated)
        
        actions_hooked = actions_fixed.detach()
        actions_fixed.register_hook(lambda x: invert_grad(actions_hooked, x))
        states_expanded = states.unsqueeze(1)   # N*1*state_dim
        q_unbounded = self.q(states_expanded, actions_fixed)   # N*K_fix*1
        grad_q_action_fixed = torch.autograd.grad(
            q_unbounded, 
            actions_fixed, 
            grad_outputs=torch.ones(q_unbounded.size(), device=self.policy.device)
            )[0]   # N*K_fix*action_dim

        self.actor_opt.zero_grad()
        actor_loss = -actions_updated
        actor_loss.backward(gradient=grad_q_action_fixed)
        self.actor_opt.step()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )


def invert_grad(actions, x):
    invert_max_v = 10.0
    greater_idx = actions > 1.0
    lower_idx = actions < -1.0
    x[greater_idx] = -invert_max_v
    x[lower_idx] = invert_max_v
    return x

