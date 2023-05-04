# python3
# Author: Scc_hy
# Create Date: 2023-05-04
# Func: TD3
# ============================================================================
import typing as typ
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base_net import TD3ValueNet as valueNet
from ._base_net import DT3PolicyNet as policyNet


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)
        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a



class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.concat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.concat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:
    def __init__(
        self,
        state_dim: int, 
        actor_hidden_layers_dim: typ.List, 
        critic_hidden_layers_dim: typ.List, 
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        TD3_kwargs: typ.Dict,
        device: torch.device
    ):
        self.actor = policyNet(
            state_dim, 
            actor_hidden_layers_dim, 
            action_dim, 
            action_bound = TD3_kwargs['action_bound']
        )
        self.critic = valueNet(
            state_dim + action_dim, 
            critic_hidden_layers_dim
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim
        self.action_low = TD3_kwargs.get('action_low', -1.0)
        self.action_high = TD3_kwargs.get('action_high', 1.0)
        self.tau = TD3_kwargs.get('tau', 0.8)
        # Normal sigma
        self.max_action = TD3_kwargs['action_bound']
        self.policy_noise = 0.2 * self.max_action 
        self.noise_clip = 0.5 * self.max_action 

        self.Q_batchsize = TD3_kwargs.get('Q_batchsize', 256) 
        self.delay_counter = -1
        self.delay_freq = 1
        self.train = False
        self.expl_noise = TD3_kwargs.get('expl_noise', 0.25)
        self.train_noise = self.expl_noise

    @torch.no_grad()
    def smooth_action(self, state):
        act_target = self.target_actor(state)
        noise = (torch.randn(act_target.shape).float() *
                self.policy_noise).clip(-self.noise_clip, self.noise_clip)
        smoothed_target_a = (act_target + noise).clip(-self.max_action, self.max_action)
        return smoothed_target_a

    @torch.no_grad()
    def policy(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        act = self.actor(state)
        if self.train:
            action_noise = np.random.normal(loc=0, scale=self.max_action * self.train_noise, size=self.action_dim)
            self.train_noise *= 0.999
            return (act.detach().numpy()[0] + action_noise).clip(-self.action_low, self.action_high)
        
        return act.detach().numpy()[0]

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )

    def update(self, samples):
        self.delay_counter += 1
        state, action, reward, next_state, done = zip(*samples)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        # 计算目标Q
        smooth_act = self.smooth_action(state)
        target_Q1, target_Q2 = self.target_critic(next_state, smooth_act)
        target_Q = torch.minimum(target_Q1, target_Q2)
        target_Q = reward + (1.0 - done) * self.gamma * target_Q
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(state, action)
        # td error
        q_loss = F.mse_loss(current_Q1.float(), target_Q.float().detach()) + F.mse_loss(current_Q2.float(), target_Q.float().detach())
        # Optimize the q_critic
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()
        if self.delay_counter == self.delay_freq:
            # actor 延迟update
            ac_action = self.actor(state)
            actor_loss = -torch.mean(self.critic.Q1(state, ac_action))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)
            self.delay_counter = -1

    def save(self, file_path):
        act_f = os.path.join(file_path, 'TD3_actor.ckpt')
        critic_f = os.path.join(file_path, 'TD3_actor.ckpt')
        torch.save(self.actor.state_dict(), act_f)
        torch.save(self.q_critic.state_dict(), critic_f)


    def load(self, file_path):
        act_f = os.path.join(file_path, 'TD3_actor.ckpt')
        critic_f = os.path.join(file_path, 'TD3_actor.ckpt')
        self.actor.set_state_dict(torch.load(act_f))
        self.q_critic.set_state_dict(torch.load(critic_f))

