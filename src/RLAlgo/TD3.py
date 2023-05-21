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


class TD3CNNValueNet(nn.Module):
    """
    输入[state, cation], 输出value
    """
    def __init__(self, pic_shape: tuple, action_dim: int, cnn_feature_dim: int):
        super(TD3CNNValueNet, self).__init__()
        pic_h, pic_w, pic_channel = pic_shape
        k_size = 12
        s_size = 12
        for _ in range(2):
            pic_h = (pic_h + 2 - 4) // 2  + 1
            pic_w = (pic_w + 2 - 4) // 2  + 1


        h_size = (pic_h-k_size) // s_size  + 1
        w_size = (pic_w-k_size) // s_size  + 1
        self.features_cnn_q1 = nn.Sequential(
            # (96+2-4) // 2 + 1
            self.conv_bn_lrelu(pic_channel, cnn_feature_dim), # output -> (batch, cnn_feature_dim, 32, 32)
            # (48+2-4) // 2 + 1
            self.conv_bn_lrelu(cnn_feature_dim, cnn_feature_dim * 2), 
            nn.AvgPool2d(kernel_size=k_size, stride=s_size)
        )
        self.features_nn_q1 = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Tanh()
        )
        self.head_q1 = nn.Linear(cnn_feature_dim * 2 * h_size * w_size + action_dim , 1)
        self.features_cnn_q2 = nn.Sequential(
            # (96+2-4) // 2 + 1
            self.conv_bn_lrelu(pic_channel, cnn_feature_dim), # output -> (batch, cnn_feature_dim, 32, 32)
            # (48+2-4) // 2 + 1
            self.conv_bn_lrelu(cnn_feature_dim, cnn_feature_dim * 2), 
            nn.AvgPool2d(kernel_size=k_size, stride=s_size)
        )
        self.features_nn_q2 = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Tanh()
        )
        self.head_q2 = nn.Linear(cnn_feature_dim * 2 * h_size * w_size + action_dim , 1)

    def forward(self, state, action):
        state_x = torch.Tensor(state).float().permute(0, 3, 1, 2)
        action_x = torch.Tensor(action).float()
        
        q1_cnnf = self.features_cnn_q1(state_x) 
        q1_cnnf = q1_cnnf.view(state_x.size(0), -1)
        q1_nnf = self.features_nn_q1(action_x)
        x_c1 = torch.cat([q1_cnnf, q1_nnf], dim=1)
        
        q2_cnnf = self.features_cnn_q2(state_x) 
        q2_cnnf = q2_cnnf.view(state_x.size(0), -1)
        q2_nnf = self.features_nn_q2(action_x)
        x_c2 = torch.cat([q2_cnnf, q2_nnf], dim=1)
        return self.head_q1(x_c1), self.head_q2(x_c2)

    def Q1(self, state, action):
        state_x = torch.Tensor(state).float().permute(0, 3, 1, 2)
        action_x = torch.Tensor(action).float()
        
        q1_cnnf = self.features_cnn_q1(state_x) 
        q1_cnnf = q1_cnnf.view(state_x.size(0), -1)
        q1_nnf = self.features_nn_q1(action_x)
        x_c1 = torch.cat([q1_cnnf, q1_nnf], dim=1)
        return self.head_q1(x_c1)

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
        

# 网络权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DT3CNNPolicyNet(nn.Module):
    """
    输入state, 输出action
    """
    def __init__(self, pic_shape: tuple, action_dim: int, cnn_feature_dim: int, action_bound: float=1.0):
        super(DT3CNNPolicyNet, self).__init__()
        pic_h, pic_w, pic_channel = pic_shape
        self.action_bound = action_bound
        k_size = 12
        s_size = 12
        for _ in range(2):
            pic_h = (pic_h + 2 - 4) // 2  + 1
            pic_w = (pic_w + 2 - 4) // 2  + 1


        h_size = (pic_h-k_size) // s_size  + 1
        w_size = (pic_w-k_size) // s_size  + 1
        self.l1 = nn.Sequential(
            # (96+2-4) // 2 + 1
            self.conv_bn_lrelu(pic_channel, cnn_feature_dim), # output -> (batch, cnn_feature_dim, 32, 32)
            # (48+2-4) // 2 + 1
            self.conv_bn_lrelu(cnn_feature_dim, cnn_feature_dim*2), 
            nn.AvgPool2d(kernel_size=k_size, stride=s_size)
        )
        self.fc_out = nn.Linear(cnn_feature_dim * 2 * h_size * w_size, action_dim)
        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        y = self.l1(x)
        y = y.view(x.size(0), -1)
        return torch.tanh(self.fc_out(y)) * self.action_bound * self.action_bound

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )


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
        device: torch.device=None
    ):
        """
        state_dim (int): 环境的sate维度  
        actor_hidden_layers_dim (typ.List): actor hidden layer 维度  
        critic_hidden_layers_dim (typ.List): critic hidden layer 维度  
        action_dim (int): action的维度  
        actor_lr (float): actor学习率  
        critic_lr (float): critic学习率  
        gamma (float): 折扣率  
        TD3_kwargs (typ.Dict): TD3算法的三个trick的输入  
            example:  
                TD3_kwargs={  
                    'action_low': env.action_space.low[0],  
                    'action_high': env.action_space.high[0],  
                - soft update parameters  
                    'tau': 0.005,   
                - trick2: Target Policy Smoothing  
                    'delay_freq': 1,  
                - trick3: Target Policy Smoothing  
                    'policy_noise': 0.2,  
                    'policy_noise_clip': 0.5,  
                - exploration noise  
                    'expl_noise': 0.25,  
                    -  探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t  
                    'expl_noise_exp_reduce_factor': 0.999  
                }  
        device (torch.device): 运行的device  
        """
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.action_low = TD3_kwargs.get('action_low', -1.0)
        self.action_high = TD3_kwargs.get('action_high', 1.0)
        self.max_action = max(abs(self.action_low), abs(self.action_high))
        self.CNN_env_flag = TD3_kwargs.get("CNN_env_flag", 0)
        if self.CNN_env_flag:
            self.actor = DT3CNNPolicyNet(
                pic_shape=TD3_kwargs['pic_shape'],
                action_dim=action_dim, 
                cnn_feature_dim=TD3_kwargs.get('cnn_feature_dim', 6),
                action_bound = self.max_action
            )
            self.critic = TD3CNNValueNet(
                pic_shape=TD3_kwargs['pic_shape'],
                action_dim=action_dim, 
                cnn_feature_dim=TD3_kwargs.get('cnn_feature_dim', 6)
            )
        else:
            self.actor = policyNet(
                state_dim, 
                actor_hidden_layers_dim, 
                action_dim, 
                action_bound = self.max_action
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
        self.action_dim = action_dim
        
        self.tau = TD3_kwargs.get('tau', 0.01)
        self.policy_noise = TD3_kwargs.get('policy_noise', 0.2) * self.max_action 
        self.policy_noise_clip = TD3_kwargs.get('policy_noise_clip', 0.5) * self.max_action 
        self.expl_noise = TD3_kwargs.get('expl_noise', 0.25)
        self.expl_noise_exp_reduce_factor = TD3_kwargs.get('expl_noise_exp_reduce_factor', 1)
        self.delay_counter = -1
        # actor延迟更新的频率: 论文建议critic更新2次， actor更新1次， 即延迟1次
        self.delay_freq = TD3_kwargs.get('delay_freq', 1)
        
        # Normal sigma
        self.train = False
        self.train_noise = self.expl_noise

    @torch.no_grad()
    def smooth_action(self, state):
        """
        trick3: Target Policy Smoothing
            在target-actor输出的action中增加noise
        """
        act_target = self.target_actor(state).cpu()
        noise = (torch.randn(act_target.shape).float() *
                self.policy_noise).clip(-self.policy_noise_clip, self.policy_noise_clip)
        smoothed_target_a = (act_target + noise).clip(self.action_low, self.action_high)
        return smoothed_target_a

    @torch.no_grad()
    def policy(self, state):
        state = torch.FloatTensor(state[np.newaxis, ...]).to(self.device)
        if self.CNN_env_flag:
            state /= 255.0
        act = self.actor(state)
        if self.train:
            action_noise = np.random.normal(loc=0, scale=self.max_action * self.train_noise, size=self.action_dim)
            self.train_noise *= self.expl_noise_exp_reduce_factor
            return (act.detach().cpu().numpy()[0] + action_noise).clip(self.action_low, self.action_high)
        
        return act.detach().cpu().numpy()[0]


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )

    def update(self, samples):
        self.delay_counter += 1
        state, action, reward, next_state, done = zip(*samples)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.tensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)
        if self.CNN_env_flag:
            state /= 255.0
            next_state /= 255.0
        
        # 计算目标Q
        smooth_act = self.smooth_action(next_state).to(self.device)
        # trick1: **Clipped Double Q-learning**: critic中有两个`Q-net`, 每次产出2个Q值，使用其中小的
        target_Q1, target_Q2 = self.target_critic(next_state, smooth_act)
        target_Q = torch.minimum(target_Q1, target_Q2)
        target_Q = reward + (1.0 - done) * self.gamma * target_Q
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(state, action)
        q_loss = F.mse_loss(current_Q1.float(), target_Q.float().detach()) + F.mse_loss(current_Q2.float(), target_Q.float().detach())
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()
        
        # trick2: **Delayed Policy Update**: actor的更新频率要小于critic(当前的actor参数可以产出更多样本)。
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

