# python3
# Create Date: 2022-10-30
# Author: Scc_hy
# Func: QLearning 
# =====================================================================================
from gym.envs.registration import register
import typing as typ
from rich.console import Console
cs = Console()
from simple_grid import DrunkenWalkEnv
from agent_utils import Agent, QTablePlot
from tqdm import tqdm
import gym


class QLearningAgent(Agent):
    def __init__(self, env: typ.ClassVar, explore_rate: float=0.1):
        super().__init__(env, explore_rate)
    
    def update(self, s: int, a: int, reward: typ.Union[int, float], n_state: int, ucb_flag: bool=False):
        estimated = self.Q[int(s), int(a)]
        # if not ucb_flag:
        gain = reward + self.gamma * max(self.Q[n_state, :])
        td = gain - estimated
        self.Q[int(s), int(a)] += self.lr_alpha * td
        # else:
            # self.Q[int(s), int(a)] += 1 / self.ucb_sa_visit_cnt_arr[int(s), int(a)] * (reward - estimated)

    def train(self, 
            epoches: int=100, 
            lr_alpha: float=0.1, 
            gamma: float=0.9, 
            render: bool=False, 
            policy_method: typ.AnyStr='TS',
            epoche_len: int=500
            ):
        """train an agent
        Args:
            epoches (int, optional): 训练游戏次数. Defaults to 100.
            lr_alpha (float, optional): 学习率. Defaults to 0.1.
            gamma (float, optional): _description_. Defaults to 0.9.
            render (bool, optional): 是否print出所有的step. Defaults to False.
            policy_method (typ.AnyStr, optional): 智能体行动策略 [
                'random', # 随机
                'greedy', # 贪心
                'softmax', # softmax
                'TS',  # 汤普森采样
                'ucb'  # 置信上界
                ] Defaults to 'TS'.
            epoche_len (int, optional): 一局游戏的长度. Defaults to 500.
        """
        self.epoche_len = epoche_len
        self.lr_alpha = lr_alpha
        self.gamma = gamma
        tq_bar = tqdm(range(epoches))
        cnt = 0
        for ep in tq_bar:
            total_reward = 0
            tq_bar.set_description(f'[ epoch {ep} ] |')
            s = self.env.reset()
            done = False
            for t in range(1, int(epoche_len + 1)):
                if done:
                    self.log(total_reward)
                    break
                cnt += 1
                if render:
                    self.env.render()
                a = self.policy(s, policy_method)
                n_state, reward, done, info = self.env.step(a)
                total_reward += reward
                self.update(s, a, reward, n_state, ucb_flag=policy_method == 'ucb')
                s = n_state


            tq_bar.set_postfix(reward=f'{reward:.3f}')
            tq_bar.update()



def main_train(env, env_name, policy_method_list):
    print(env.nrow, env.ncol)
    final_play_res = []
    for method in policy_method_list:
        ql_ = QLearningAgent(env, explore_rate=0.01)
        ql_.train(
            epoches=1000, 
            lr_alpha=0.1, 
            gamma=0.9, 
            render=False, 
            policy_method=method,
            epoche_len=500)
        ql_.smooth_plot(window_size=50, freq=1, title=f'{method} | ')
        res = ql_.play(render=False, method=method)
        final_play_res.append(res)


        ploter = QTablePlot(ql_)
        ploter.plot(title=f'{env_name}-{method} | ')
    return final_play_res


if __name__ == "__main__":
    # theAlley, walkInThePark
    env_name = 'theAlley'
    env = DrunkenWalkEnv(map_name=env_name)
    gym_env_name = 'FrozenLakeEasy-v0'
    register(id=gym_env_name, entry_point="gym.envs.toy_text:FrozenLakeEnv",
            kwargs={"is_slippery": False})
    gym_env = gym.make(gym_env_name)
    policy_method_list = ['random', 'greedy' ,'softmax', 'TS', 'ucb']
    final_play_res = main_train(env, env_name, policy_method_list)
    cs.print(final_play_res)

