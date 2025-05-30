

import typing as typ
import numpy as np
import pandas as pd
from collections import deque
import random


class replayBuffer:
    def __init__(self, capacity: int, np_save: bool=False):
        self.buffer = deque(maxlen=capacity)
        self.np_save = np_save
    
    def add(self, state, action, reward, next_state, done):
        if self.np_save:
            self.buffer.append( (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) )
        else:
            self.buffer.append( (state, action, reward, next_state, done) )

    def add_more(self, *args):
        self.buffer.append( args )

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int) -> deque:
        samples = random.sample(self.buffer, batch_size)
        return samples
    

class rolloutReplayBuffer(replayBuffer):
    def __init__(self, capacity: int, np_save: bool=False, gamma=0.99):
        super().__init__(capacity, np_save)
        self.one_traj = []
        self.traj_weight = []
        self.gamma = gamma 
        
    def add_more(self, *args):
        # s, a, r, n_s, done, log_prob, R
        self.one_traj.append([*args, 0])
        if args[4]:
            self.trajectory_advantage()
            self.buffer.append( self.one_traj )
            self.one_traj = []
        
    def __len__(self):
        return sum(len(tj) for tj in self.buffer)

    def sample(self, batch_size: int) -> deque:
        buffer_ = []
        w = []
        for idx, tj in enumerate(self.buffer):
            buffer_.extend(tj)
            w.extend([self.traj_weight[idx]] * len(tj))
            
        samples = np.random.choice(np.arange(len(buffer_)), batch_size, p=self.softmax_w(w))
        return [buffer_[i] for i in samples]
    
    def softmax_w(self, w):
        p_ = np.array(w)
        return np.exp(p_) / np.exp(p_).sum()

    def trajectory_advantage(self):
        k_step = len(self.one_traj)
        r = 0
        for t in reversed(range(k_step)):
            r = self.one_traj[t][2] + self.gamma * r * (1 - int(self.one_traj[t][4]))
            self.one_traj[t][-1] = r
        # adv = R - self.critic(state[0])
        
        self.traj_weight.append(r)
