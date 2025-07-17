# python3
# Author: Scc_hy
# Create Date: 2025-06-23
# Func: CQL
#       refernce: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779)
#                  GitHub  https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
# ============================================================================

import typing as typ
import numpy as np
import os
import math
from gymnasium.wrappers.normalize import RunningMeanStd
import torch
from torch import nn
from torch.nn import functional as F
from .._base_net import DDPGValueNet as valueNet
from .._base_net import SACPolicyNet as policyNet
from copy import deepcopy


class CQL_H_SAC:
    """
    SAC + CQL(H)
    Reference
    ----------
    - Github: https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
    - paper: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779)
    """
    def __init__(
        self, 
        state_dim: int,
        actor_hidden_layers_dim: typ.List[int],
        critic_lr: float,
        action_dim: int,
        critic_hidden_layers_dim: typ.List[int],
        actor_lr: float,
        gamma: float,
        CQL_kwargs: typ.Dict=dict(
            temp=1.0,
            min_q_weight=1.0,
            num_random=10,
            tau=0.05
        ),
        device: torch.device='cpu',
        alpha_lr: float=None,
        reward_func: typ.Callable=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device 
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.reward_func = reward_func
        self.bc_flag = CQL_kwargs.get('bc_flag', False)
        self.norm_obs = CQL_kwargs.get('norm_obs', False)
        if alpha_lr is None:
            self.alpha_lr = actor_lr
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, action_bound=CQL_kwargs.get('action_bound', 1.0)).to(device)
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

        self.gamma = gamma
        self.device = device
        self.action_bound = CQL_kwargs.get('action_bound', 1.0)
        self.tau = CQL_kwargs.get('tau', 0.05)
        self.target_entropy = CQL_kwargs['target_entropy']
        self.reward_scale = CQL_kwargs.get('reward_scale', 1)
        ## min Q
        self.temp = CQL_kwargs.get('temp', 1)
        self.min_q_weight = CQL_kwargs.get('min_q_weight', 1)
        self.num_random = CQL_kwargs.get('num_random', 10)

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        
        # NormalizeObservation
        self.obs_rms = RunningMeanStd(shape=(state_dim,))

    def normalize(self, obs, update_flag=True):
        if not self.norm_obs:
            return obs
        if update_flag:
            self.obs_rms.update(obs if isinstance(obs, np.ndarray) else obs.detach().cpu().numpy())
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-6)

    def _critic_to_device(self, device):
        self.critic_1 = self.critic_1.to(device)
        self.critic_2 = self.critic_2.to(device)
        self.target_critic_1 = self.target_critic_1.to(device)
        self.target_critic_2 = self.target_critic_2.to(device)

    def policy(self, state):
        state = torch.FloatTensor(np.array([self.normalize(state, False)])).to(self.device)
        action = self.actor(state, retrun_log_prob=not self.bc_flag)[0]
        return action.cpu().detach().numpy()[0]

    def _get_tensor_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

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

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
    
    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def bc_update(self, batch_tensor, wandb_w=None):
        state, action, reward, next_state, done = batch_tensor
        state = self.normalize(state, True)
        action = action.float().to(self.device)
        state = state.float().to(self.device) 
        # 1- actor_loss 
        new_act, log_prob = self.actor(state, retrun_log_prob=False)
        imitation_loss = torch.mean(0.5 * (action - new_act) ** 2)
        self.actor_opt.zero_grad()
        imitation_loss.backward()
        self.actor_opt.step()

        if wandb_w is not None:
            log_dict = {                
                'imitation_loss': imitation_loss.detach()
            }
            wandb_w.log(log_dict)
        return -1

    def update(self, batch_tensor, wandb_w=None):
        state, action, reward, next_state, done = batch_tensor
        state = self.normalize(state, True)
        state = state.float().to(self.device)
        action = action.float().to(self.device)
        reward = reward.to(self.device)
        if self.reward_func is not None:
            reward = self.reward_func(reward)
        reward = reward * self.reward_scale 
        next_state = self.normalize(next_state, False)
        next_state = next_state.float().to(self.device)
        done = done.to(self.device)

        # 1- actor_loss 
        new_act, log_prob = self.actor(state)
        q1_v = self.target_critic_1(state, new_act)
        q2_v = self.target_critic_2(state, new_act)
        #   Theorem 3.5  eqution  5
        #       D_{CQL}(\pi, \pi_\beta) = \sum_a \pi(a|s) \cdot (\pi(a|a)/\pi_beta(a|s) - 1)
        #            https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/policies.py#L21
        #            log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value);
        #            self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)
        #            log_prob = log_prob.sum(dim=1, keepdim=True)
        actor_loss = torch.mean(-(torch.min(q1_v, q2_v) - self.log_alpha.exp().detach() * log_prob))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # 2- crtic loss 
        q1 = self.critic_1(state, action).float()
        q2 = self.critic_2(state, action).float()
        td_target = self.calc_target(reward, next_state, done)
        critic_1_loss = 0.5 * torch.mean((q1 - td_target.float().detach())**2)
        critic_2_loss = 0.5 * torch.mean((q2 - td_target.float().detach())**2)

        # add CQL 进行采样
        # \mu sampling a - state
        state_temp = state.unsqueeze(1).repeat(1, self.num_random, 1).view(
            state.shape[0] * self.num_random, state.shape[1])
        cur_act, cur_log_proba = self.actor(state_temp)
        cur_log_proba = cur_log_proba.view(state.shape[0], self.num_random, 1)
        # \mu sampling a - next state
        next_state_temp = next_state.unsqueeze(1).repeat(1, self.num_random, 1).view(
            next_state.shape[0] * self.num_random, next_state.shape[1])
        next_act, next_log_proba = self.actor(next_state_temp)
        next_log_proba = next_log_proba.view(next_state.shape[0], self.num_random, 1) 
        # uniform distribution
        random_act_tensor = torch.FloatTensor(q2.shape[0] * self.num_random, action.shape[-1]).uniform_(
            -self.action_bound, self.action_bound).to(self.device)

        q1_rand =  self._get_tensor_values(state, random_act_tensor, self.critic_1)
        q2_rand =  self._get_tensor_values(state, random_act_tensor, self.critic_2)
        q1_curr_actions = self._get_tensor_values(state, cur_act, network=self.critic_1)
        q2_curr_actions = self._get_tensor_values(state, cur_act, network=self.critic_2)
        q1_next_actions = self._get_tensor_values(state, next_act, network=self.critic_1)
        q2_next_actions = self._get_tensor_values(state, next_act, network=self.critic_2)
        
        # cat_q1 = torch.cat(
        #     [q1_rand, q1.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        # )
        # cat_q2 = torch.cat(
        #     [q2_rand, q2.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        # )
        # std_q1 = torch.std(cat_q1, dim=1)
        # std_q2 = torch.std(cat_q2, dim=1)

        random_h = cur_act.shape[-1] * np.log(2)
        # print(f'{q1_rand.shape=} {q1_next_actions.shape=} {next_act.shape=} {next_log_proba.shape=}, {q1_curr_actions.shape=} {cur_log_proba.shape=}')
        cat_q1 = torch.cat([
            q1_rand + random_h, 
            q1_next_actions - next_log_proba.detach(), 
            q1_curr_actions - cur_log_proba.detach()
        ], 1)
        cat_q2 = torch.cat([
            q2_rand + random_h, 
            q2_next_actions - next_log_proba.detach(), 
            q2_curr_actions - cur_log_proba.detach()
        ], 1)
        # eqution4 CQL(H)
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.temp
        # Subtract the log likelihood of data
        min_qf1_loss = (min_qf1_loss - q1.mean()) * self.min_q_weight
        min_qf2_loss = (min_qf2_loss - q2.mean()) * self.min_q_weight
        
        critic_1_loss += min_qf1_loss
        critic_2_loss += min_qf2_loss
        self.critic_1_opt.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        critic_2_loss.backward(retain_graph=True)
        self.critic_2_opt.step()

        # update alpha
        alpha_loss = torch.mean((-log_prob - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        # update target_critic 
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        if wandb_w is not None:
            log_dict = {
                "mean_q": torch.min(q1_v, q2_v).detach().cpu().numpy().mean()/self.reward_scale,
                'min_qf1_loss': min_qf1_loss.detach(),
                'min_qf2_loss': min_qf2_loss.detach(),
                'critic_1_loss': critic_1_loss.detach(),
                'critic_2_loss': critic_2_loss.detach(),
                'alpha_loss': alpha_loss.detach(),
                'actor_loss': actor_loss.detach(),
            }
            wandb_w.log(log_dict)
        return torch.min(q1_v, q2_v).detach().cpu().numpy().mean()/self.reward_scale

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        act_f = os.path.join(file_path, 'cql_actor.ckpt')
        critic_f = os.path.join(file_path, 'cql_critic.ckpt')
        if self.norm_obs:
            self.actor.obs_mean.data = torch.tensor(self.obs_rms.mean).to(self.device)
            self.actor.obs_var.data = torch.tensor(self.obs_rms.var).to(self.device)
        
        torch.save(self.actor.state_dict(), act_f)
        torch.save(self.critic_1.state_dict(), critic_f)

    def load_model(self, file_path):
        act_f = os.path.join(file_path, 'cql_actor.ckpt')
        critic_f = os.path.join(file_path, 'cql_critic.ckpt')
        actor_d = torch.load(act_f, map_location='cpu')
        self.actor.load_state_dict(actor_d)
        self.critic_1.load_state_dict(torch.load(critic_f, map_location='cpu'))
        self.critic_2.load_state_dict(torch.load(critic_f, map_location='cpu'))
        self.target_critic_1.load_state_dict(torch.load(critic_f, map_location='cpu'))
        self.target_critic_2.load_state_dict(torch.load(critic_f, map_location='cpu'))
        
        self.obs_rms.mean = actor_d['obs_mean'].detach().cpu().numpy()
        self.obs_rms.var = actor_d['obs_var'].detach().cpu().numpy()
        
        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)





def mdn_loss(mu, logvar, logpi, a):
    var = torch.exp(logvar.clamp(-10, 10))
    log_prob = -0.5 * ((a - mu) ** 2 / var + logvar + math.log(2 * math.pi))
    log_mix = torch.logsumexp(logpi + log_prob, dim=-1)
    return -log_mix.mean()

