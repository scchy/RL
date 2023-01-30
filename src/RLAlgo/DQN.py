

import typing as typ
import numpy as np
import pandas as pd
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from ._base_net import VANet, QNet
import copy


class DQN:
    """
    仅支持离散action
    """
    def __init__(self, 
                state_dim: int,
                hidden_layers_dim: typ.List[int],
                action_dim: int,
                learning_rate: float,
                gamma: float,
                epsilon: float=0.05,
                target_update_freq: int=1,
                device: typ.AnyStr='cpu',
                dqn_type: typ.AnyStr='DQN'
                ):
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':
            self.q = VANet(state_dim, hidden_layers_dim, action_dim).to(device)
        else:
            self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(device)

        self.target_q = copy.deepcopy(self.q)
        self.cost_func = nn.MSELoss()
        self.opt = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.target_q.to(device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device
        self.dqn_type = dqn_type
        self.count = 0

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        action = self.target_q(torch.FloatTensor(state))
        return np.argmax(action.detach().numpy())

    def update(self, samples: deque):
        self.count += 1
        # sample -> tensor
        states, actions, rewards, next_states, done = zip(*samples)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.Tensor(np.array(actions)).view(-1, 1).to(self.device)
        rewards = torch.Tensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        done = torch.Tensor(np.array(done)).view(-1, 1).to(self.device)
        
        Q_s = self.q(states)
        Q1_s = self.target_q(next_states)
        # print(Q_s.shape, actions.shape)
        Q_sa = Q_s.gather(1, actions.long())
        # 下一状态最大值
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            max_Q1sa = Q1_s.gather(1, max_action)
        else:
            max_Q1sa = Q1_s.max(1)[0].view(-1, 1)
        
        Q_target = rewards + self.gamma * max_Q1sa * (1 - done)
        # MSELoss
        loss = self.cost_func(Q_sa.float(), Q_target.float())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.count % self.target_update_freq == 0:
            self.target_q.load_state_dict(
                self.q.state_dict()
            )
