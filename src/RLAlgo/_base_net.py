# python3
# Create Date: 2023-01-22
# Func: QNet 
# =================================================================================

import typing as typ
import numpy as np
import pandas as pd
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Normal


class VANet(nn.Module):
    # 可以让智能体开始关注不同动作优势值的差异
    # Dueling DQN 能够更加频繁、准确地学习状态价值函数
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(VANet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            })) 
        
        # 表示采取不同动作的差异性
        self.adv_header = nn.Linear(hidden_layers_dim[-1], action_dim)
        # 状态价值函数
        self.v_header = nn.Linear(hidden_layers_dim[-1], 1)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        
        adv = self.adv_header(x)
        v = self.v_header(x)
        # Q值由V值和A值计算得到
        Q = v + adv - adv.mean().view(-1, 1)  
        return Q

    def complete(self, lr):
        self.cost_func = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)
    
    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()



class QNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(QNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            }))
        self.head = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)

    def complete(self, lr):
        self.cost_func = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)
    
    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()



class SACPolicyNet(nn.Module):
    """
    输入state, 输出action 和 -熵
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, action_bound: float=1.0):
        super(SACPolicyNet, self).__init__()
        self.action_bound = action_bound
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))
        self.fc_mu = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layers_dim[-1], action_dim)


    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        
        mean_ = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 3e-5
        dist = Normal(mean_, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return  action * self.action_bound, log_prob



class DDPGPolicyNet(nn.Module):
    """
    输入state, 输出action
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, action_bound: float=1.0):
        super(DDPGPolicyNet, self).__init__()
        self.action_bound = action_bound
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))

        self.fc_out = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))

        return torch.tanh(self.fc_out(x)) * self.action_bound



class PPOPolicyNet(nn.Module):
    """
    continuity action:
    normal distribution (mean, std) 
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(PPOPolicyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))
        self.fc_mu = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layers_dim[-1], action_dim)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        
        mean_ = torch.tanh(self.fc_mu(x))
        # np.log(1 + np.exp(2))
        std = F.softplus(self.fc_std(x)) + 1e-5
        return mean_, std



class DDPGValueNet(nn.Module):
    """
    输入[state, cation], 输出value
    """
    def __init__(self, state_action_dim: int, hidden_layers_dim: typ.List):
        super(DDPGValueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_action_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1).float() # 拼接状态和动作
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x) 
    
    
    
class PPOValueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim):
        super(PPOValueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)
        
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x)
