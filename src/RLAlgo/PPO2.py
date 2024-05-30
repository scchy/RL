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
from ._base_net import PPOValueCNN, PPOPolicyCNN
from .grad_ana import gradCollecter
import copy
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import deque
from typing import List


def mini_batch(batch, mini_batch_size, adv_norm=False):
    states, actions, old_log_probs, adv, td_target, b_values = zip(*batch)
    rand_ids = np.random.randint(0, len(states), mini_batch_size)
    adv = torch.stack(adv)
    if adv_norm:
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)
    return torch.stack(states)[rand_ids, ...], torch.stack(actions)[rand_ids, :], \
        torch.stack(old_log_probs).reshape(-1, 1)[rand_ids, :], adv[rand_ids, :], \
        torch.stack(td_target)[rand_ids, :], torch.stack(b_values).reshape(-1, 1)[rand_ids, :]


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
        self.act_type = act_type
        act_attention_flag = PPO_kwargs.get('act_attention_flag', False)
        self.dist_type = dist_type
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.cnn_flag = PPO_kwargs.get('cnn_flag', False)
        self.continue_action_flag = PPO_kwargs.get('continue_action_flag', True)
        self.max_pooling = PPO_kwargs.get('max_pooling', True)
        if self.cnn_flag:
            self.actor = PPOPolicyCNN(state_dim, actor_hidden_layers_dim, action_dim, dist_type=self.dist_type, act_type=self.act_type, continue_action_flag=self.continue_action_flag, max_pooling=self.max_pooling).to(device)
            self.critic = PPOValueCNN(state_dim, critic_hidden_layers_dim, act_type=self.act_type, max_pooling=self.max_pooling).to(device)
        else:
            self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, dist_type=self.dist_type, act_type=self.act_type).to(device)
            self.critic = valueNet(state_dim, critic_hidden_layers_dim, act_type=self.act_type).to(device)
        self.opt = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr, "eps": 1e-5}, 
            {'params': self.critic.parameters(), 'lr': actor_lr, "eps": 1e-5}
        ])

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
        self.min_adv_norm = PPO_kwargs.get('min_adv_norm', False)
        self.min_batch_collate_func = partial(
            mini_batch, 
            mini_batch_size=self.minibatch_size,
            adv_norm=self.min_adv_norm 
        )
        self.update_cnt = 0
        self.anneal_lr = PPO_kwargs.get('act_type', False)
        self.max_grad_norm = PPO_kwargs.get('max_grad_norm', 0)
        if self.anneal_lr:
            self.num_iters = PPO_kwargs['num_episode']

        self.clip_vloss = PPO_kwargs.get('clip_vloss', False)
        

    def _action_fix(self, act):
        if not self.continue_action_flag:
            return act
        if self.dist_type == 'beta':
            # beta 0-1 -> low ~ high
            return act * (self.action_high - self.action_low) + self.action_low
        return act
    
    def _action_return(self, act):
        if not self.continue_action_flag:
            return act.reshape(-1)
        if self.dist_type == 'beta' and self.continue_action_flag:
            # low ~ high -> 0-1 
            act_out = (act - self.action_low) / (self.action_high - self.action_low)
            return (act_out * 1 + 0).clip(1e-4, 9.999)
        return act

    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action_dist = self.actor.get_dist(state, self.action_bound)
        action = action_dist.sample()
        action = self._action_fix(action)
        # print(f"action.cpu().detach().numpy()={action.cpu().detach().numpy()}")
        out = action.cpu().detach().numpy()
        if len(out.shape) == 3:
            return out[0]
        return out

    @torch.no_grad()
    def data_prepare(self, samples: deque):
        state, action, reward, next_state, done = zip(*samples)
        # print(f"state[0]={state[0].shape} ") # 4, 3
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = np.stack(reward) # n, env_nums
        # state=torch.Size([2048, 4, 3]), action=torch.Size([2048, 4, 1]), reward=(2048, 4)
        if self.reward_func is not None:
            reward = self.reward_func(reward)

        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        #print(f'next_state.shape={next_state.shape}')
        done = np.stack(done)
        td_target = np.zeros_like(reward) 
        advantage = np.zeros_like(reward) 
        old_v_arr = np.zeros_like(reward) 
        for env_idx in range(state.size(1)):
            # compute adv 
            old_v = self.critic(state[:, env_idx, ...]).detach().cpu().numpy().flatten()
            last_v = self.critic(next_state[:, env_idx, ...]).detach().cpu().numpy()[len(reward)-1]
            # print(f'old_v={old_v.shape} last_v={last_v.shape}')
            gae = 0
            for t in reversed(range(len(reward))):
                if t == len(reward) - 1:
                    mask = 1.0 - done[t, env_idx]
                    next_v = last_v
                else:
                    mask = 1.0 - done[t+1, env_idx]
                    next_v = old_v[t+1]
                
                delta = reward[t, env_idx] + self.gamma * next_v * mask - old_v[t]
                gae = delta + self.gamma * self.lmbda * mask * gae
                advantage[t, env_idx] = gae
            # recompute
            td_target[:, env_idx] = advantage[:, env_idx] + old_v
            old_v_arr[:, env_idx] = old_v

        advantage = advantage.reshape(-1, 1)
        td_target = td_target.reshape(-1, 1)
        old_v_arr = old_v_arr.reshape(-1, 1)
        try:
            b, env_nums, action_dim = action.size()
            action = action.reshape(-1, action_dim)
        except Exception as e:
            b, env_nums = action.size()
            action_dim = 1
            action = action.reshape(-1, 1)
            
        if self.cnn_flag:
            b, env_nums, channel, s1, s2 = state.size()
            state = state.reshape(-1, channel, s1, s2)
        else:
            _, _, state_dim = state.size()
            state = state.reshape(-1, state_dim)
        action_dists = self.actor.get_dist(state, self.action_bound)
        if action_dim == 1:
            old_log_probs = action_dists.log_prob(self._action_return(action))
        else:
            old_log_probs = action_dists.log_prob(self._action_return(action)).sum(1)

        # trick1: batch-normalize https://zhuanlan.zhihu.com/p/512327050  
        if not self.min_adv_norm:
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        return state, action, old_log_probs, \
                torch.FloatTensor(advantage).to(self.device), \
                torch.FloatTensor(td_target).to(self.device), \
                torch.FloatTensor(old_v_arr).to(self.device)

    def lr_update(self, opt, lr, iteration):
        # total_timesteps = cfg.num_episode * 100
        # num_iters = args.total_timesteps // cfg.off_buffer_size # batch_size
        frac = max(1e-8, 1.0 - (iteration - 1.0) / self.num_iters)
        opt.param_groups[0]["lr"] = frac * lr
        opt.param_groups[1]["lr"] = frac * lr

    def update(self, samples_buffer: deque, wandb = None):
        self.update_cnt += 1
        state, action, old_log_probs, advantage, td_target, b_values = self.data_prepare(samples_buffer)
        # print(f"update old_log_probs={old_log_probs.shape}")
        # print(f'{state.shape=} {action.shape=} {advantage.shape=} {td_target.shape=} {b_values.shape=}')
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
        # train_loader = ppo_train_iter(
        #     self.minibatch_size,
        #     state, action, old_log_probs, advantage, td_target, b_values
        # )
        for _ in range(self.k_epochs):
            for state_, action_, old_log_prob, adv, td_v, before_v in train_loader:
                self.opt.zero_grad()
                action_dists = self.actor.get_dist(state_, self.action_bound)
                new_log_prob = action_dists.log_prob(self._action_return(action_))
                if self.continue_action_flag:
                    new_log_prob = new_log_prob.sum(1).reshape(-1, 1)
                else:
                    new_log_prob = new_log_prob.reshape(-1, 1)
                try:
                    entropy_loss = action_dists.entropy().sum(1).mean()
                except Exception as e:
                    # sperate action
                    entropy_loss = action_dists.entropy().sum().mean()
                # e(log(a/b))
                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean().float() - self.ent_coef * entropy_loss
                new_v = self.critic(state_).float()
                td_v = td_v.detach().float()
                # print(
                #     f'{state_.shape=} {action_.shape=} {old_log_prob.shape=} {new_log_prob.shape=} {ratio.shape=} {surr1.shape=} {surr2.shape=}',
                #     f'{new_v.shape=} {td_v.shape=}'
                # )
                if self.clip_vloss:
                    # ref: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                    v = torch.clamp(new_v, before_v - self.eps, before_v + self.eps)
                    # print(f'v={v.shape} td_v={td_v.shape} new_v={new_v.shape}')
                    critic_loss = 0.5 * torch.mean(
                        torch.max((v - td_v).pow(2), (new_v - td_v).pow(2))
                    )
                else:
                    critic_loss = 0.5 * torch.mean((new_v - td_v).pow(2))
                
                total_loss = actor_loss + self.critic_coef * critic_loss
                loss_item = total_loss.cpu().detach().numpy()
                if np.isnan(loss_item):
                    print(f'>>>>>>>>>> nan loss=', loss_item)
                    print(f'>>>>>>>>>> actor_loss=', actor_loss)
                    print(f'>>>>>>>>>> ratio =', ratio)
                    print(f'>>>>>>>>>> new_log_prob =', new_log_prob)
                    print(f'>>>>>>>>>> old_log_prob =', old_log_prob)
                    print(f'>>>>>>>>>> adv =', adv)
                    print(f'>>>>>>>>>> critic_loss=', critic_loss)

                total_loss.backward()
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
                self.opt.step()
        
        if self.anneal_lr:
            self.lr_update(self.opt, self.actor_lr, self.update_cnt)
            # self.lr_update(self.critic_opt, self.critic_lr, self.update_cnt)
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
            self.actor.load_state_dict(torch.load(act_f))
            self.critic.load_state_dict(torch.load(critic_f))
        except Exception as e:
            self.actor.load_state_dict(torch.load(act_f, map_location='cpu'))
            self.critic.load_state_dict(torch.load(critic_f, map_location='cpu'))

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.opt = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.actor_lr, "eps": 1e-5}, 
            {'params': self.critic.parameters(), 'lr': self.actor_lr, "eps": 1e-5}
        ])
        self.update_cnt = 0

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
