# python3
# Create Date: 2021-12-21
# Author: Scc_hy
# Func: 智能体
# ================================================================

import gym
from gym.envs.registration import register
import numpy as np
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})
env = gym.make("FrozenLakeEasy-v0")


class Agent:
    def __init__(self, env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.n_actions = env.action_space.n
        self.actions = list(range(self.n_actions))
        self.Q = self._make_Q_table()
        self.recorder = []
    
    def policy(self, state):
        """
        \epsilon - 探索
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        if state in self.Q and sum(self.Q[state]) != 0:
            return np.argmax(self.Q[state])
        return np.random.randint(self.n_actions)

    def _make_Q_table(self):
        total_state = self.env.observation_space.n
        Q = {i:[0.0] * self.n_actions for i in range(total_state)}
        return Q

    def record_reset(self):
        self.recorder = []

    def record_reward(self, r):
        self.recorder.append(r)
    
    def print_info(self, epoch, interval=50):
        mean_ = np.mean(self.recorder[-interval:])
        std_ = np.std(self.recorder[-interval:])
        epoch_str = str(epoch).zfill(3)
        print(f'[ Epoch:{epoch_str} ] rewards {mean_:.3f} (+/- {std_:.3f})')

    def plot_reward(self, interval=50):
        import matplotlib.pyplot as plt
        plot_r = []
        for i in range(len(self.recorder)- interval):
            plot_r.append(np.mean(self.recorder[i:i+interval]))
        plt.plot(plot_r)
        plt.show()
