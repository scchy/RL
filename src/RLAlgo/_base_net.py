# python3
# Create Date: 2023-01-22
# Func: QNet 
# reference: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
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
import gymnasium as gym


class CNNQNet(nn.Module):
    """
    输入[state, cation], 输出value
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(CNNQNet, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten()
        )
        # self.cnn_feature = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2, 0),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2, 1, 0),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        self.features = nn.ModuleList()
        cnn_out_dim = self._get_cnn_out_dim()
        print(f'state_dim => {state_dim}')
        print(f'cnn_out_dim => {cnn_out_dim}')
        self.ln = nn.LayerNorm(cnn_out_dim)
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else cnn_out_dim, h),
                'linear_activation': nn.ReLU()
            }))

        self.head = nn.Linear(hidden_layers_dim[-1], action_dim)

    @torch.no_grad
    def _get_cnn_out_dim(self):
        pic = torch.randn((1, 4, self.# The above code is declaring a variable named "state_dim" in
        # Python. However, the code is incomplete and does not assign a
        # value to the variable.
        state_dim, self.state_dim))
        return self.cnn_feature(pic).shape[1]
    
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        try:
            x = self.cnn_feature(state)
        except Exception as e:
            state = state.permute(0, 3, 1, 2)
            x = self.cnn_feature(state)

        x = self.ln(x)
        for layer1 in self.features:
            x = layer1['linear_activation'](layer1['linear'](x))

        return self.head(x)


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
        std = F.softplus(self.fc_std(x).clip(-100, 650)) + 3e-5
        dist = Normal(mean_, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        if len(log_prob.shape) >= 2:
            log_prob = log_prob.sum(1, keepdim=True)
        return  action * self.action_bound, log_prob


class DDPGPolicyNet(nn.Module):
    """
    输入state, 输出action
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, action_bound: float=1.0, state_feature_share=False):
        super(DDPGPolicyNet, self).__init__()
        self.state_feature_share = state_feature_share
        if state_feature_share:
            state_dim = 512
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


class TD3PolicyNet(nn.Module):
    """
    输入state, 输出action
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, action_bound: float=1.0, state_feature_share: bool=False):
        super(TD3PolicyNet, self).__init__()
        self.state_feature_share = state_feature_share
        if state_feature_share:
            state_dim = 512
        self.action_bound = action_bound
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.Tanh()
            }))

        self.fc_out = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))

        return torch.tanh(self.fc_out(x)) * self.action_bound


class TD3CNNPolicyNet(nn.Module):
    """
    输入state, 输出action
    """
    def __init__(self, 
                state_dim: int, 
                hidden_layers_dim: typ.List, 
                action_dim: int, 
                action_bound: typ.Union[float, gym.Env]=1.0, 
                state_feature_share: bool=False
                ):
        super(TD3CNNPolicyNet, self).__init__()
        self.state_feature_share = state_feature_share
        self.low_high_flag = hasattr(action_bound, "action_space")
        print('action_bound=',action_bound)
        self.action_bound = action_bound
        if self.low_high_flag:
            self.action_high = torch.FloatTensor(action_bound.action_space.low)
            self.action_low = torch.FloatTensor(action_bound.action_space.high)
        # self.cnn_feature = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten()
        )
        self.cnn_out_ln = nn.LayerNorm([512])
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else 512, h),
                'linear_action': nn.ReLU()
            }))
        
        self.fc_out = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.final_ln = nn.LayerNorm([action_dim])
        # self._init_w()
    
    def _init_w(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                p = nn.init.xavier_normal_(p)
        
    def max_min_scale(self, act):
        """
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        """
        # print("max_min_scale(", act, ")")
        device_ = act.device
        action_range = self.action_high.to(device_) - self.action_low.to(device_)
        act_std = (act - -1.0) / 2.0
        return act_std * action_range.to(device_) + self.action_low.to(device_)

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        try:
            x = self.cnn_feature(state)
        except Exception as e:
            print(state.shape)
            state = state.permute(0, 3, 1, 2)
            x = self.cnn_feature(state)

        x = self.cnn_out_ln(x)
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))

        device_ = x.device
        if self.low_high_flag:
            return self.max_min_scale(torch.tanh(self.final_ln(self.fc_out(x))))
        return torch.tanh(self.final_ln(self.fc_out(x)).clip(-6.0, 6.0)) * self.action_bound



class PPOPolicyBetaNet(nn.Module):
    """
    continuity action:
    normal distribution (mean, std) 
    Chou P W, Maturana D, Scherer S. Improving stochastic policy gradients in continuous control with deep reinforcement learning using the beta distribution[C]
    //International conference on machine learning. PMLR, 2017: 834-843.
    https://proceedings.mlr.press/v70/chou17a.html
    my trick: action attention
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, 
                 state_feature_share: bool=False,
                 dist_type='beta', 
                 act_type='relu',
                 act_attention_flag=False):
        super(PPOPolicyBetaNet, self).__init__()
        if state_feature_share:
            state_dim = 512
        self.act_attention_flag = act_attention_flag
        print(f'PPOPolicyBetaNet.act_attention_flag={self.act_attention_flag}')
        self.dist_type = dist_type
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True) if act_type == 'relu' else nn.Tanh()
            }))
        self.alpha_layer = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.beta_layer = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.norm_out_layer = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.__init()
        # attention 
        self.alpha_attn_ly = nn.Linear(action_dim, action_dim)
        self.alpha_act_tran_ly = nn.Linear(action_dim, action_dim)
        
        self.beta_attn_ly = nn.Linear(action_dim, action_dim)
        self.beta_act_tran_ly = nn.Linear(action_dim, action_dim)
        
        self.mean_attn_ly = nn.Linear(action_dim, action_dim)
        self.mean_act_tran_ly = nn.Linear(action_dim, action_dim)
        self.sft = nn.Softmax(dim=1)

    def simple_attention(self, a, act_tran_ly, ln):
        a = act_tran_ly(torch.tanh(a))
        one = torch.ones_like(a).to(a.device)
        a_one = torch.einsum('bc,ba->bca', a, one).float()
        return torch.sum(a_one * self.sft(ln(a_one)),  1)

    def forward(self, x):
        # nan -> pip install --upgrade numpy
        x_org = x
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        
        alpha = self.alpha_layer(x)
        beta = self.beta_layer(x)
        mean = self.norm_out_layer(x)
        if self.act_attention_flag:
            alpha = self.simple_attention(alpha, self.alpha_act_tran_ly, self.alpha_attn_ly)
            beta = self.simple_attention(beta, self.beta_act_tran_ly, self.beta_attn_ly)
            mean = self.simple_attention(mean, self.mean_act_tran_ly, self.mean_attn_ly)
    
        alpha = F.softplus(alpha.clip(-100, 650)) + 1.0
        beta = F.softplus(beta.clip(-100, 650)) + 1.0
        
        if np.isnan(alpha.sum().cpu().detach().numpy()):
            print('x=', x)
            print('x_org=', x_org)
        return alpha, beta, mean

    def mean(self, s):
        alpha, beta, mean = self.forward(s)
        if self.dist_type == 'beta':
            mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean
    
    def get_dist(self, x, max_action=1.0):
        alpha, beta, mean = self.forward(x)
        if self.dist_type == "beta":
            dist = torch.distributions.Beta(alpha, beta)
            return dist

        mean = mean * max_action
        log_std = self.log_std.expand_as(mean) 
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = torch.distributions.Normal(mean, std)
        return dist

    def __init(self):
        for layer in self.features:
            orthogonal_init(layer['linear'], gain=np.sqrt(2))
            
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)
        orthogonal_init(self.norm_out_layer, gain=0.01)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

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


class TD3ValueNet(nn.Module):
    """
    输入[state, cation], 输出value
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_layers_dim: typ.List, 
                 state_feature_share=False
                ):
        super(TD3ValueNet, self).__init__()
        state_action_dim = state_dim + action_dim
        self.state_feature_share = state_feature_share
        if state_feature_share:
            state_action_dim = 512 + action_dim
        self.features_q1 = nn.ModuleList()
        self.features_q2 = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features_q1.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_action_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
            self.features_q2.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_action_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
            
        self.head_q1 = nn.Linear(hidden_layers_dim[-1] , 1)
        self.head_q2 = nn.Linear(hidden_layers_dim[-1] , 1)
        
    def forward(self, state, action):
        # if self.state_feature_share:
        #     state = self.feat_extrator(state)
        x = torch.cat([state, action], dim=1).float() # 拼接状态和动作
        x1 = x
        x2 = x
        for layer1, layer2 in zip(self.features_q1, self.features_q2):
            x1 = layer1['linear_activation'](layer1['linear'](x1))
            x2 = layer2['linear_activation'](layer2['linear'](x2))
        return self.head_q1(x1), self.head_q2(x2)

    def Q1(self, state, action):
        # if self.state_feature_share:
        #     state = self.feat_extrator(state)
        x = torch.cat([state, action], dim=1).float() # 拼接状态和动作
        for layer in self.features_q1:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head_q1(x) 


class TD3CNNValueNet(nn.Module):
    """
    输入[state, cation], 输出value
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_layers_dim: typ.List, 
                 state_feature_share=False
                ):
        super(TD3CNNValueNet, self).__init__()
        self.state_feature_share = state_feature_share
        self.q1_cnn_feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten()
        )
        self.q2_cnn_feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten()
        )
        self.features_q1 = nn.ModuleList()
        self.features_q2 = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim + [action_dim]):
            self.features_q1.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else 512, h),
                'linear_activation': nn.ReLU()
            }))
            self.features_q2.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else 512, h),
                'linear_activation': nn.ReLU()
            }))

        self.act_q1_fc = nn.Linear(action_dim, action_dim)
        self.act_q2_fc = nn.Linear(action_dim, action_dim)

        self.head_q1_bf = nn.Linear(action_dim * 2, action_dim)
        self.head_q2_bf = nn.Linear(action_dim * 2, action_dim)
        
        self.head_q1 = nn.Linear(action_dim, 1)
        self.head_q2 = nn.Linear(action_dim, 1)
        # self.apply(weights_init)
        # self._init_w()
    
    def _init_w(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                p = nn.init.xavier_normal_(p)
        
    def forward(self, state, action):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        try:
            x1 = self.q1_cnn_feature(state)
            x2 = self.q2_cnn_feature(state)
        except Exception as e:
            state = state.permute(0, 3, 1, 2)
            x1 = self.q1_cnn_feature(state)
            x2 = self.q2_cnn_feature(state)
            
        for layer1, layer2 in zip(self.features_q1, self.features_q2):
            x1 = layer1['linear_activation'](layer1['linear'](x1))
            x2 = layer2['linear_activation'](layer2['linear'](x2))

        # 拼接状态和动作
        act1 = torch.relu(self.act_q1_fc(action.float()))
        act2 = torch.relu(self.act_q2_fc(action.float()))
        x1 = torch.relu( self.head_q1_bf(torch.cat([x1, act1], dim=-1).float()))
        # print("torch.cat([x1, action], dim=-1)=", torch.cat([x1, act1], dim=-1)[:5, :])
        x2 = torch.relu( self.head_q2_bf(torch.cat([x2, act2], dim=-1).float()))
        return self.head_q1(x1), self.head_q2(x2)

    def Q1(self, state, action):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        try:
            x = self.q1_cnn_feature(state)
        except Exception as e:
            state = state.permute(0, 3, 1, 2)
            x = self.q1_cnn_feature(state)

        for layer in self.features_q1:
            x = layer['linear_activation'](layer['linear'](x))

        # 拼接状态和动作
        act1 = torch.relu(self.act_q1_fc(action.float()))
        x = torch.relu( self.head_q1_bf(torch.cat([x, act1], dim=-1).float()))
        return self.head_q1(x) 


class PPOValueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, act_type='relu'):
        super(PPOValueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True) if act_type == 'relu' else nn.Tanh()
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)
        self.__init()

    def __init(self):
        for layer in self.features:
            orthogonal_init(layer['linear'], gain=np.sqrt(2))

        orthogonal_init(self.head, gain=1)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x)



class stateFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(stateFeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512)
        )
        # self.apply(weights_init)
    
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        try:
            out = self.feature(state)
        except Exception as e:
            print(state.shape)
            state = state.permute(0, 3, 1, 2)
            out = self.feature(state)
        return out.view(out.size(0), -1)


# 网络权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
