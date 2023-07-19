# python3
# Author: Scc_hy
# Create Date: 2023-01-25
# Func: SAC
# ============================================================================
import typing as typ
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ._base_net import DDPGValueNet as valueNet
from ._base_net import SACPolicyNet as policyNet
from copy import deepcopy


class SAC:
    """
    一个策略网络
    两个价值网络
    两个目标网络
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
        SAC_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, action_bound=SAC_kwargs.get('action_bound', 1.0)).to(device)
        self.critic_1 = valueNet(state_dim+action_dim, critic_hidden_layers_dim)
        self.critic_2 = valueNet(state_dim+action_dim, critic_hidden_layers_dim)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self._critic_to_device(device)
        
        # optim
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.gamma = gamma
        self.device = device
        self.tau = SAC_kwargs.get('tau', 0.05)
        self.target_entropy = SAC_kwargs['target_entropy']
            
    def _critic_to_device(self, device):
        self.critic_1 = self.critic_1.to(device)
        self.critic_2 = self.critic_2.to(device)
        self.target_critic_1 = self.target_critic_1.to(device)
        self.target_critic_2 = self.target_critic_2.to(device)
    
    def policy(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = self.actor(state)[0]
        return action.cpu().detach().numpy()[0]

    def calc_target(self, reward, next_state, dones):
        # 计算目标Q值
        next_actions, log_prob = self.actor(next_state)
        # entropy = -log_prob
        q1_v = self.target_critic_1(next_state, next_actions)
        q2_v = self.target_critic_2(next_state, next_actions)
        next_v = torch.min(q1_v, q2_v) - self.log_alpha.exp() * log_prob
        return reward + self.gamma * next_v * ( 1 - dones )


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )


    def update(self, samples):
        state, action, reward, next_state, done = zip(*samples)


        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        reward = (reward + 10.0) / 10.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        td_target = self.calc_target(reward, next_state, done)
        # update critic net
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(state, action).float(), td_target.float().detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(state, action).float(), td_target.float().detach())
        )
        self.critic_1_opt.zero_grad()
        critic_1_loss.backward()
        self.critic_1_opt.step()
        self.critic_2_opt.zero_grad()
        critic_2_loss.backward()
        self.critic_2_opt.step()
        
        # update actor
        new_act, log_prob = self.actor(state)
        q1_v = self.target_critic_1(next_state, new_act)
        q2_v = self.target_critic_2(next_state, new_act)
        actor_loss = torch.mean(self.log_alpha.exp() * log_prob - torch.min(q1_v, q2_v))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
    
        # update alpha
        alpha_loss = torch.mean((-log_prob - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
