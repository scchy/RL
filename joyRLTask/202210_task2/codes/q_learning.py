# python3
# Create Date: 2022-10-30
# Author: Scc_hy
# Func: QLearning 
# =====================================================================================

import typing as typ
import os
import sys
import numpy  as np
from rich.console import Console
import matplotlib.pyplot as plt
cs = Console()
from simple_grid import DrunkenWalkEnv
from tqdm import tqdm
plt.style.use('ggplot')


class AlleyAgent:
    def __init__(self, env: typ.ClassVar, lr_alpha: float=0.1, gamma: float=0.9, explore_rate: float=0.2):
        self.env = env
        self.gamma = gamma
        self.lr_alpha = lr_alpha
        self.explore_rate = explore_rate
        self.Q = np.array([])
        self.__init_QTable()
    
    def __init_QTable(self):
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.Q = np.random.random((n_states, n_actions))
        # s ="{}".format(["Left","Down","Right","Up"][action_index])
        # self.Q[:, [1, -1]] = 0.0
        
    def __softmax(self, actions_v):
        return np.exp(actions_v) / np.exp(actions_v).sum()
    
    def _softmax_policy(self, s):
        return np.random.choice(np.arange(len(self.Q[s])), size=1, p=self.__softmax(self.Q[s]))[0]

    def _ThompsonSampling_policy(self, s):
        # s ="{}".format(["Left","Down","Right","Up"][action_index])
        if np.random.random() < self.explore_rate:
            return np.random.randint(len(self.Q[s]))
            # return np.random.choice([0, 2])
        return np.argmax(self.Q[s])

    def _ucb_policy(self, s):
        return 

    def policy(self, s, method='TS'):
        if method == 'TS':
            return self._ThompsonSampling_policy(s)
        if method == 'softmax':
            return self._softmax_policy(s)
        if method == 'ucb':
            return self._softmax_policy(s)
        return self._ThompsonSampling_policy(s)
    
    def _actions2str(self, action_list):
        str_ = ''
        for act in action_list:
            str_ += self.env.action_to_string(act) + ' -> '
        return str_[:-4]

    def play(self, render=True):
        s = self.env.reset()
        action_list = []
        step_cnt = 0
        done = False
        while not done:
            step_cnt += 1
            if render:
                self.env.render()
            a = self.policy(s)
            action_list.append(a)
            n_state, reward, done, info = env.step(a)
            s = n_state
        
        print(f'[ TOTAL-STEP: {step_cnt} ]| {self._actions2str(action_list)}')



class QLearning(AlleyAgent):
    def __init__(self, env, explore_rate=0.1):
        super().__init__(env, explore_rate)
        self.records = []
        self.round_records = []
        self.early_stop = False
    
    def dp_transate(self):
        return 
    
    def train(self, 
            epoches=100, 
            lr_alpha=0.1, 
            gamma=0.9, 
            theta=0.005,
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
                n_state, reward, done, info = env.step(a)

                gain = reward + self.gamma * max(self.Q[n_state])
                estimated = self.Q[int(s), int(a)]
                td = gain - estimated
                self.Q[int(s), int(a)] += self.lr_alpha * td
                s = n_state
            else:
                self.log(reward)

            tq_bar.set_postfix(reward=f'{reward:.3f}')
            tq_bar.update()
            

    def log(self, reward):
        self.records.append(reward)
        self.round_records.append(reward)
    
    def smooth_plot(self, window_size=10, freq=1, title=''):
        record_arr = np.array(self.records)
        record_smooth = []
        std_list = []
        if self.early_stop and  len(record_arr) < window_size:
            window_size = 10
        for i in range(1, len(self.records), freq):
            tmp_arr = record_arr[max(0, i-window_size):i]
            record_smooth.append(np.mean(tmp_arr))
            std_list.append(np.std(tmp_arr))
            
        plt.title(f'{title}Learning Rewards Trend')
        plt.plot(record_smooth, label=f'Rewards for each {window_size} episode.')
        plt.fill_between(
            x=np.arange(len(record_smooth)),
            y1=np.array(record_smooth) - np.array(std_list), 
            y2=np.array(record_smooth) + np.array(std_list), 
            color='green', alpha=0.1
            )
        # plt.plot(self.round_records, label='Finished-Rounds')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # theAlley, walkInThePark
    env = DrunkenWalkEnv(map_name="walkInThePark")
    # ag = Agent(env)
    # s = env.reset()
    # a = ag.policy(s)
    # n_state, reward, done, info = env.step(a)
    for method in  ['TS', 'softmax']:
        ql_ = QLearning(env, explore_rate=0.05)
        ql_.train(
            epoches=1000, 
            lr_alpha=0.1, 
            gamma=0.9, 
            theta=0.005,
            render=False, policy_method=method)
        ql_.smooth_plot(window_size=50, freq=1, title=f'{method} | ')
        ql_.play()


    print(ql_.Q.shape)
    print(ql_.Q)
