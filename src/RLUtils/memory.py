

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

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int) -> deque:
        samples = random.sample(self.buffer, batch_size)
        return samples