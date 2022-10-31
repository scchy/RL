# python3
# Create Date: 2022-10-31
# Author: Scc_hy
# Func: Agent
# =====================================================================================

import typing as typ
import numpy  as np
from rich.console import Console
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from enum import Enum
cs = Console()
plt.rcParams['font.sans-serif'] =['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class Agent:
    def __init__(self, env: typ.ClassVar, lr_alpha: float=0.1, gamma: float=0.9, explore_rate: float=0.2):
        self.env = env
        self.gamma = gamma
        self.lr_alpha = lr_alpha
        self.explore_rate = explore_rate
        self.Q = np.array([])
        self.__init_QTable()
        self.records = []
        self.round_records = []
    
    def __init_QTable(self):
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.Q = np.zeros((n_states, n_actions)) # np.random.random((n_states, n_actions)) / 100
        # s ="{}".format(["Left","Down","Right","Up"][action_index])
        # self.Q[:, [1, -1]] = 0.0
        
    def __softmax(self, actions_v):
        return np.exp(actions_v + 1e-5 ) / np.sum(np.exp(actions_v + 1e-5), axis=0)
    
    def _softmax_policy(self, s):
        return np.random.choice(range(len(self.Q[s])), size=1, p=self.__softmax(self.Q[s]))[0]

    def _ThompsonSampling_policy(self, s):
        # s ="{}".format(["Left","Down","Right","Up"][action_index])
        if np.random.random() < self.explore_rate:
            return np.random.randint(len(self.Q[s]))
        if sum(self.Q[s]) != 0:
            return np.argmax(self.Q[s])
        return np.random.randint(len(self.Q[s]))

    def _ucb_policy(self, s):
        return 

    def policy(self, s, method='TS'):
        if method == 'TS':
            return self._ThompsonSampling_policy(s)
        if method == 'softmax':
            return self._softmax_policy(s)
        if method == 'ucb':
            return self._ucb_policy(s)
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
            n_state, reward, done, info = self.env.step(a)
            s = n_state
        # print(f'[ TOTAL-STEP: {step_cnt} ]| {self._actions2str(action_list)}')

    def log(self, reward):
        self.records.append(reward)
        self.round_records.append(reward)
    
    def smooth_plot(self, window_size=10, freq=1, title=''):
        record_arr = np.array(self.records)
        record_smooth = []
        std_list = []
        if len(record_arr) < window_size:
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
        plt.legend()
        plt.show()



class Direct(Enum):
    Left_v = 0
    Down_v = 1
    Right_v = 2
    Up_v = 3
    Left = -1
    Down = -1
    Right = +1
    Up = +1
    Left_s = r'$\Leftarrow$'
    Down_s = r'$\Downarrow$'
    Right_s = r'$\Rightarrow$'
    Up_s = r'$\Uparrow$'




class QTablePlot:
    def __init__(self, agent):
        self.agent = agent
        self.env_rows, self.env_cols = agent.env.nrow, agent.env.ncol
        self.table = np.zeros((agent.env.nrow * 3, agent.env.ncol * 3))
        self.direct = Direct
        self.text_record_dict = {}
        self.record_SHG = {}
    
    def _fill_table(self, q_table):
        text_str_list = [
            Direct.Left_s.value, 
            Direct.Down_s.value, 
            Direct.Right_s.value, 
            Direct.Up_s.value
        ]

        env_desc_str = ''.join(''.join(i) for i in self.agent.env.desc.astype(str))
        for r in range(self.env_rows):
            for c in range(self.env_cols):
                s = r * self.env_cols + c
                center_r = 1 + (self.env_rows - r - 1) * 3
                # center_r = 1 + r * 3
                center_c = 1 + c * 3
                # "Left","Down","Right","Up"
                self.table[
                    center_r , center_c + Direct.Left.value
                ] = q_table[s, Direct.Left_v.value]
                self.table[
                    center_r , center_c + Direct.Right.value
                ] = q_table[s, Direct.Right_v.value]
                self.table[
                    center_r + Direct.Down.value , center_c
                ] = q_table[s, Direct.Down_v.value]
                self.table[
                    center_r + Direct.Up.value , center_c
                ] = q_table[s, Direct.Up_v.value]
                # center
                self.table[
                    center_r , center_c
                ] = q_table[s].mean()

                idx = np.argmax(q_table[s])
                name = text_str_list[idx]
                mv_v = Direct[Direct(idx).name[:-2]].value

                if idx in [0, 2]:
                    self.text_record_dict[f'{center_r},{center_c + mv_v}'] = name

                else:
                    self.text_record_dict[f'{center_r + mv_v},{center_c}'] = name
                
                if env_desc_str[s] != '.' and env_desc_str[s] != 'F':
                    self.record_SHG[f'{center_r},{center_c}'] = env_desc_str[s]


    def plot(self, title=''):
        self._fill_table(self.agent.Q)

        plt.title(f'{title}Agent Qtable')
        plt.imshow(
            self.table,
            cmap="RdBu_r", #cm.RdYlGn ,#
            interpolation="bilinear",
            vmin=-np.abs(self.table).max() , 
            vmax=np.abs(self.table).max()
        )
        plt.xlim(-0.5, 3 * self.env_cols - 0.5)
        plt.ylim(-0.5, 3 * self.env_rows - 0.5)
        plt.xticks(np.arange(-0.5, 3 * self.env_cols, 3), range(self.env_cols + 1))
        plt.yticks(np.arange(-0.5, 3 * self.env_rows, 3), range(self.env_rows + 1))
        plt.grid(True)
        for k, v in self.text_record_dict.items():
            y_, x_ = k.split(',')
            plt.text(int(x_), int(y_), f'{v}', va='center', ha='center')
        
        for k, v in self.record_SHG.items():
            y_, x_ = k.split(',')
            plt.text(int(x_), int(y_), f'{v}', va='center', ha='center', color='darkred', fontdict={'weight': 'bold'})
        plt.show()





