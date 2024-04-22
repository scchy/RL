# python3
# Author: Scc_hy
# Create Date: 2024-01-12
# Func: PPO 
#       在线收集一段数据然后，进行训练
# trick reference: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# my trick: action attention
# ============================================================================
import typing as typ
import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from torch.nn import functional as F
from ._base_net import PPOValueNet as valueNet
from ._base_net import PPOPolicyBetaNet as policyNet
from .grad_ana import gradCollecter
import copy
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import deque
from typing import List



def mini_batch(batch, mini_batch_size):
    mini_batch_size += 1
    states, actions, old_log_probs, adv, td_target, b_values = zip(*batch)
    # trick1: batch_normalize
    adv = torch.stack(adv)
    adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)
    return torch.stack(states[:mini_batch_size]), torch.stack(actions[:mini_batch_size]), \
        torch.stack(old_log_probs[:mini_batch_size]), adv[:mini_batch_size], \
            torch.stack(td_target[:mini_batch_size]), torch.stack(b_values[:mini_batch_size])


def ppo_train_iter(mini_batch_size: int, states: tensor, actions: tensor, old_log_probs: tensor, 
                 advantage: tensor, td_target: tensor, b_values: tensor):
    batch_size = states.size(0)
    b_inds = np.arange(batch_size)
    # adv = torch.stack(advantage)
    for start in range(0, batch_size, mini_batch_size):
        np.random.shuffle(b_inds)
        end = start + mini_batch_size
        rand_ids = b_inds[start:end]
        yield states[rand_ids, :], actions[rand_ids, :], \
            old_log_probs[rand_ids, :], advantage[rand_ids, :], \
            td_target[rand_ids, :], b_values[rand_ids, :]


class memDataset(Dataset):
    def __init__(self, states: tensor, actions: tensor, old_log_probs: tensor, 
                 advantage: tensor, td_target: tensor, b_values: tensor):
        super(memDataset, self).__init__()
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantage = advantage
        self.td_target = td_target
        self.b_values = b_values
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        old_log_probs = self.old_log_probs[index]
        adv = self.advantage[index]
        td_target = self.td_target[index]
        b_value = self.b_values[index]
        return states, actions, old_log_probs, adv, td_target, b_value


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



class PPO2:
    """
    PPO2算法, 采用截断方式
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
        self.grad_collector = gradCollecter()
        dist_type = PPO_kwargs.get('dist_type', 'beta')
        act_type = PPO_kwargs.get('act_type', 'relu')
        act_attention_flag = PPO_kwargs.get('act_attention_flag', False)
        self.dist_type = dist_type
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, dist_type=dist_type, act_type=act_type, act_attention_flag=act_attention_flag).to(device)
        self.critic = valueNet(state_dim, critic_hidden_layers_dim, act_type=act_type).to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.k_epochs = PPO_kwargs['k_epochs'] # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps'] # PPO中截断范围的参数
        self.sgd_batch_size = PPO_kwargs.get('sgd_batch_size', 512)
        self.minibatch_size = PPO_kwargs.get('minibatch_size', 128)
        self.action_bound = PPO_kwargs.get('action_bound', 1.0)
        self.action_low = torch.FloatTensor([-1 * self.action_bound]).to(device)
        self.action_high = torch.FloatTensor([self.action_bound]).to(device)
        if 'action_space' in PPO_kwargs:
            self.action_low = torch.FloatTensor(PPO_kwargs['action_space'].low).to(device)
            self.action_high = torch.FloatTensor(PPO_kwargs['action_space'].high).to(device)
            print("action_space=", self.action_low , '->', self.action_high)
        
        self.ent_coef = PPO_kwargs.get('ent_coef', 0.0)
        self.critic_coef = PPO_kwargs.get('critic_coef', 0.5)
        self.count = 0 
        self.device = device
        self.reward_func = reward_func
        self.min_batch_collate_func = partial(mini_batch, mini_batch_size=self.minibatch_size)
        self.update_cnt = 0
        self.anneal_lr = PPO_kwargs.get('act_type', False)
        self.max_grad_norm = PPO_kwargs.get('max_grad_norm', 0)
        if self.anneal_lr:
            self.num_iters = PPO_kwargs['num_episode'] * 100 // PPO_kwargs['off_buffer_size']

        self.clip_vloss = PPO_kwargs.get('clip_vloss', False)


    def _action_fix(self, act):
        if self.dist_type == 'beta':
            # beta 0-1 -> low ~ high
            return act * (self.action_high - self.action_low) + self.action_low
        return act 
    
    def _action_return(self, act):
        if self.dist_type == 'beta':
            # low ~ high -> 0-1 
            act_out = (act - self.action_low) / (self.action_high - self.action_low)
            return (act_out * 1 + 0).clip(1e-4, 9.999)
        return act 

    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action_dist = self.actor.get_dist(state, self.action_bound)
        action = action_dist.sample()
        action = self._action_fix(action)
        return action.cpu().detach().numpy()[0]


    def _one_deque_pp(self, samples: deque):
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
        action_dists = self.actor.get_dist(state, self.action_bound)
        old_log_probs = action_dists.log_prob(self._action_return(action))
        return state, action, old_log_probs, advantage, td_target

    def data_prepare(self, samples_list: List[deque]):
        state_pt_list = []
        action_pt_list = []
        old_log_probs_pt_list = []
        advantage_pt_list = []
        td_target_pt_list = []
        for sample in samples_list:
            state_i, action_i, old_log_probs_i, advantage_i, td_target_i = self._one_deque_pp(sample)
            state_pt_list.append(state_i)
            action_pt_list.append(action_i)
            old_log_probs_pt_list.append(old_log_probs_i)
            advantage_pt_list.append(advantage_i)
            td_target_pt_list.append(td_target_i)

        state = torch.concat(state_pt_list) 
        action = torch.concat(action_pt_list) 
        old_log_probs = torch.concat(old_log_probs_pt_list) 
        advantage = torch.concat(advantage_pt_list) 
        td_target = torch.concat(td_target_pt_list)
        return state, action, old_log_probs, advantage, td_target

    def lr_update(self, opt, lr, iteration):
        # total_timesteps = cfg.num_episode * 100
        # num_iters = args.total_timesteps // cfg.off_buffer_size # batch_size
        frac = max(1e-6, 1.0 - (iteration - 1.0) / self.num_iters)
        opt.param_groups[0]["lr"] = frac * lr

    def update(self, samples_list: List[deque], wandb = None):
        self.update_cnt += 1
        state, action, old_log_probs, advantage, td_target = self.data_prepare(samples_list)
        b_values = self.critic(state).detach().reshape(-1)
        if len(old_log_probs.shape) == 2:
            old_log_probs = old_log_probs.sum(dim=1)

        d_set = memDataset(state, action, old_log_probs, advantage, td_target, b_values)
        train_loader = DataLoader(
            d_set,
            batch_size=self.sgd_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.min_batch_collate_func
        )

        for _ in range(self.k_epochs):
            for state_, action_, old_log_prob, adv, td_v, before_v in train_loader:
                action_dists = self.actor.get_dist(state_, self.action_bound)
                log_prob = action_dists.log_prob(self._action_return(action_))
                entropy = action_dists.entropy().sum(1).mean()
                if len(log_prob.shape) == 2:
                    log_prob = log_prob.sum(dim=1)
                new_log_prob = action_dists.log_prob(self._action_return(action_))
                entropy_loss = action_dists.entropy().sum(1).mean()
                # e(log(a/b))
                ratio = torch.exp(log_prob - old_log_prob.detach())
                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv

                actor_loss = -torch.min(surr1, surr2).mean().float() - self.ent_coef * entropy_loss
                new_v = self.critic(state_).float()
                td_v = td_v.detach().float()
                if self.clip_vloss:
                    # ref: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                    v = torch.clamp(new_v, before_v - self.eps, before_v + self.eps)
                    critic_loss = 0.5 * torch.mean(
                        torch.max((v - td_v).pow(2), (new_v - td_v).pow(2))
                    )
                else:
                    critic_loss = 0.5 * torch.mean((new_v - td_v).pow(2))

                total_loss = actor_loss + self.critic_coef * critic_loss
                loss_item = total_loss.cpu().detach().numpy()

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.grad_collector(self.actor.parameters())
                self.grad_collector(self.critic.parameters())
                if wandb is not None:
                    wandb.log({
                        'total_loss': loss_item,
                        'actor_gard_norm': self.grad_collector.collected_grad[-2],
                        'critic_gard_norm': self.grad_collector.collected_grad[-1]
                        })
                if self.max_grad_norm > 0.0001:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) 
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm ) 
                self.actor_opt.step()
                self.critic_opt.step()

        if self.anneal_lr:
            self.lr_update(self.actor_opt, self.actor_lr, self.update_cnt)
            self.lr_update(self.critic_opt, self.critic_lr, self.update_cnt)
        return True

    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        act_f = os.path.join(file_path, 'PPO_actor.ckpt')
        critic_f = os.path.join(file_path, 'PPO_critic.ckpt')
        torch.save(self.actor.state_dict(), act_f)
        torch.save(self.critic.state_dict(), critic_f)
        
        
    def load_model(self, file_path):
        act_f = os.path.join(file_path, 'PPO_actor.ckpt')
        critic_f = os.path.join(file_path, 'PPO_critic.ckpt')
        try:
            self.actor.load_state_dict(torch.load(act_f, map_location='cpu'))
            self.critic.load_state_dict(torch.load(critic_f, map_location='cpu'))
        except Exception as e:
            self.actor.load_state_dict(torch.load(act_f, map_location='cpu'), strict=False)
            self.critic.load_state_dict(torch.load(critic_f, map_location='cpu'), strict=False)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
        self.update_cnt = 0

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
