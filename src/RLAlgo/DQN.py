

import typing as typ
import numpy as np
import pandas as pd
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from ._base_net import VANet, QNet, CNNQNet
import copy
import os

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
                dqn_type: typ.AnyStr='DQN',
                epsilon_start: float = None,
                epsilon_decay_steps: int = None
                ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_range = (epsilon_start - epsilon) if epsilon_start is not None else None
        print(f'self.epsilon_range={self.epsilon_range}')
        self.action_dim = action_dim
        if  'DuelingDQN' in dqn_type:
            self.q = VANet(state_dim, hidden_layers_dim, action_dim).to(device)
        elif "CNN" in dqn_type:
            self.q = CNNQNet(state_dim, hidden_layers_dim, action_dim).to(device)
        else: 
            self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(device)

        self.target_q = copy.deepcopy(self.q)
        self.cost_func = nn.MSELoss()
        self.learning_rate = learning_rate
        self.opt = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.target_q.to(device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device
        self.dqn_type = dqn_type
        self.count = 1
        self.training = True
        self.train()

    def train(self):
        self.training = True
        self.q.train()
        self.target_q.train()

    def eval(self):
        self.training = False
        self.q.eval()
        self.target_q.eval()
    
    def _epsilon_update(self):
        # d ln(x) = 1/x
        org_range = np.log(self.epsilon_decay_steps)
        new_range = self.epsilon_range
        reduce_ = np.log(self.count)/org_range * new_range
        ep_now = self.epsilon_start - reduce_
        if ep_now > self.epsilon_end:
            return ep_now
        return self.epsilon_end

    @torch.no_grad()
    def policy(self, state):
        if self.training and self.epsilon_range is not None:
            self.epsilon = self._epsilon_update()
        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        try:
            state = torch.FloatTensor(state[np.newaxis, ...]).to(self.device)
        except Exception as e:
            state = torch.stack(state._frames).float().to(self.device)

        action = self.q(state)
        return np.argmax(action.cpu().detach().numpy())

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
        if 'DoubleDQN' in self.dqn_type:
            # a* = argmax Q(s_{t+1}, a; w)
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            # doubleDQN Q(s_{t+1}, a*; w')
            max_Q1sa = Q1_s.gather(1, max_action)
        else:
            # simple method:  avoid bootstrapping 
            max_Q1sa = Q1_s.max(1)[0].view(-1, 1)
        
        Q_target = rewards + self.gamma * max_Q1sa * (1 - done)
        # MSELoss
        loss = self.cost_func(Q_sa.float(), Q_target.float())
        self.opt.zero_grad()
        loss.backward()
        if 'CNN' in self.dqn_type:
            for n, p in self.q.cnn_feature[0].named_parameters():
                g_sum = p.grad.sum()
                if g_sum > -0.05 and g_sum < 0.05:
                    print(f"\loss={loss:.5f} self.q.cnn_feature[0] grad.sum={g_sum:.5f}")
                break
        self.opt.step()
        if self.count % self.target_update_freq == 0:
            self.target_q.load_state_dict(
                self.q.state_dict()
            )

    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        q_f = os.path.join(file_path, 'q_net.ckpt')
        torch.save(self.q.state_dict(), q_f)

    def load_model(self, file_path):
        q_f = os.path.join(file_path, 'q_net.ckpt')
        try: 
            self.target_q.load_state_dict(torch.load(q_f))
            self.q.load_state_dict(torch.load(q_f))
        except Exception as e:
            self.target_q.load_state_dict(torch.load(q_f, map_location='cpu'))
            self.q.load_state_dict(torch.load(q_f, map_location='cpu'))

        self.q.to(self.device)
        self.target_q.to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
