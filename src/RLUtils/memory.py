

import typing as typ
import numpy as np
import pandas as pd
from collections import deque
import random


class replayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state,done) )

    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int) -> deque:
        samples = random.sample(self.buffer, batch_size)
        return samples