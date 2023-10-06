# python3
# Author: Scc_hy
# Create Date: 2023-01-25
# Func: PPO
# ============================================================================
import typing as typ
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ._base_net import PPOValueNet as valueNet
from ._base_net import PPOPolicyBetaNet as policyNet
import copy
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import deque


def mini_batch(batch, mini_batch_size):
    mini_batch_size += 1
    states, actions, old_log_probs, adv, td_target = zip(*batch)
    return torch.stack(states[:mini_batch_size]), torch.stack(actions[:mini_batch_size]), \
        torch.stack(old_log_probs[:mini_batch_size]), torch.stack(adv[:mini_batch_size]), torch.stack(td_target[:mini_batch_size])

        
class memDataset(Dataset):
    def __init__(self, states: tensor, actions: tensor, old_log_probs: tensor, 
                 advantage: tensor, td_target: tensor):
        super(memDataset, self).__init__()
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantage = advantage
        self.td_target = td_target
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        old_log_probs = self.old_log_probs[index]
        adv = self.advantage[index]
        td_target = self.td_target[index]
        return states, actions, old_log_probs, adv, td_target


def compute_advantage(gamma, lmbda, td_delta, done):
    td_delta = td_delta.cpu().detach().numpy()
    done_arr = done.cpu().detach().numpy()
    adv_list = [] 
    adv = 0
    for delta, d in zip(td_delta[::-1], done_arr[::-1]):
        adv = gamma * lmbda * adv * (1.0 - d) + delta
        adv_list.append(adv)
    adv_list.reverse()
    return torch.FloatTensor(np.array(adv_list))


class PPO:
    """
    PPO算法, 采用截断方式
    """
    def __init__(self,
                state_dim: int,
                actor_hidden_layers_dim: typ.List,
                critic_hidden_layers_dim: typ.List,
                action_dim: int,
                actor_lr: float,
                critic_lr: float,
                gamma: float,
                PPO_kwargs: typ.Dict,
                device: torch.device,
                reward_func: typ.Optional[typ.Callable]=None
                ):
        dist_type = PPO_kwargs.get('dist_type', 'beta')
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, dist_type=dist_type).to(device)
        self.critic = valueNet(state_dim, critic_hidden_layers_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.k_epochs = PPO_kwargs['k_epochs'] # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps'] # PPO中截断范围的参数
        self.sgd_batch_size = PPO_kwargs.get('sgd_batch_size', 512)
        self.minibatch_size = PPO_kwargs.get('minibatch_size', 128)
        self.action_bound = PPO_kwargs.get('action_bound', 1.0)
        self.count = 0 
        self.device = device
        self.reward_func = reward_func
        self.min_batch_collate_func = partial(mini_batch, mini_batch_size=self.minibatch_size)
    
    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action_dist = self.actor.get_dist(state, self.action_bound)
        action = action_dist.sample()
        return action.cpu().detach().numpy()[0]
        
    def update(self, samples: deque):
        state, action, reward, next_state, done = zip(*samples)

        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        if self.reward_func is not None:
            reward = self.reward_func(reward)

        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)
        
        old_v = self.critic(state)
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_delta = td_target - old_v
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta, done).to(self.device)
        # recompute
        td_target = advantage + old_v
        # trick1: batch_normalize
        advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + + 1e-5)
        action_dists = self.actor.get_dist(state, self.action_bound)
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(action)
        d_set = memDataset(state, action, old_log_probs, advantage, td_target)
        train_loader = DataLoader(
            d_set,
            batch_size=self.sgd_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.min_batch_collate_func
        )


        for _ in range(self.k_epochs):
            for state_, action_, old_log_prob, adv, td_v in train_loader:
                action_dists = self.actor.get_dist(state, self.action_bound)
                log_prob = action_dists.log_prob(action_)
                
                # e(log(a/b))
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv

                actor_loss = torch.mean(-torch.min(surr1, surr2)).float()
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(state_).float(), td_v.detach().float())
                ).float()
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) 
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) 
                self.actor_opt.step()
                self.critic_opt.step()

        return True
