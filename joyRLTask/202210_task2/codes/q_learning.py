# python3
# Create Date: 2022-10-30
# Author: Scc_hy
# Func: QLearning 
# =====================================================================================
from gym.envs.registration import register
import typing as typ
import os
import sys
import numpy  as np
from rich.console import Console
import matplotlib.pyplot as plt
cs = Console()
from simple_grid import DrunkenWalkEnv
from agent_utils import Agent, QTablePlot
from tqdm import tqdm
import gym



class QLearningAgent(Agent):
    def __init__(self, env, explore_rate=0.1):
        super().__init__(env, explore_rate)
    
    def update(self, s, a, reward, n_state):
        gain = reward + self.gamma * max(self.Q[n_state, :])
        estimated = self.Q[int(s), int(a)]
        td = gain - estimated
        self.Q[int(s), int(a)] += self.lr_alpha * td

    def train(self, 
            epoches=100, 
            lr_alpha=0.1, 
            gamma=0.9, 
            render=False, 
            policy_method='TS'):
        self.lr_alpha = lr_alpha
        self.gamma = gamma
        tq_bar = tqdm(range(epoches))
        cnt = 0
        for ep in tq_bar:
            tq_bar.set_description(f'[ epoch {ep} ] |')
            s = self.env.reset()
            done = False
            while not done:
                cnt += 1
                if render:
                    self.env.render()
                a = self.policy(s, policy_method)
                n_state, reward, done, info = self.env.step(a)
                self.update(s, a, reward, n_state)
                s = n_state
            else:
                self.log(reward)

            tq_bar.set_postfix(reward=f'{reward:.3f}')
            tq_bar.update()



if __name__ == "__main__":
    # theAlley, walkInThePark
    # env = DrunkenWalkEnv(map_name="walkInThePark")
    env = DrunkenWalkEnv(map_name="theAlley")
    # register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
    #         kwargs={"is_slippery": False})
    # env = gym.make('FrozenLakeEasy-v0')
    print(env.nrow, env.ncol)
    for method in  ['softmax', 'TS']: 
        ql_ = QLearningAgent(env, explore_rate=0.01)
        ql_.train(
            epoches=1000, 
            lr_alpha=0.1, 
            gamma=0.9, 
            render=False, policy_method=method)
        ql_.smooth_plot(window_size=50, freq=1, title=f'{method} | ')
        ql_.play()

        ploter = QTablePlot(ql_)
        ploter.plot(title=f'walkInThePark-{method} | ')

    print(ql_.Q.shape)
    print(ql_.Q)

