# python3
# Create Date:2022-12-11 ~ 13
# Author: Scc_hy
# Func:  DQN + DoubleDQN + 
# Tip: 
#     replayBuffer 一个经验回放池
#     QNet  QNet & TargetQNet
#     DQN:  dqn + doubleDQN + 
#     Config: 训练的时候的重要参数
#     Pendulum_dis_to_con: Pendulum-v0 动作空间变成连续空间
# =================================================================================

from collections import deque
import numpy as np
import random
import torch
from torch import nn
import typing as typ
import copy
import pickle
import gym
from tqdm import tqdm
import os

print(gym.__version__) # '0.26.2'


__doc__ = """
DQN算法流程:
1. 初始化QNet
2. 复制相同的参数到 TargetQNet
3. 初始化经验回放池R
4. 训练
for e=1 -> E do:
    获取环境初始状态s1
    for t=1 -> T do:
        根据QNet以e-greedy方法选择动作at
        执行at, 获得回报rt, 环境状态变为s_t+1
        将(st, at, rt, s_t+1)存储进回放池R中
        若R中的数据足够, 从R中采样N个数据 { (si, ai, ri, si+1) }+i=1,...,N 
        对每个数据, 用目标网络计算 yi = ri + gamma max(TagetQNet(si+1, a))
        最小化目标损失 L=1/N\sum (yi - QNet(si, ai))^2, 更新当前网络
        更新目标网络 (这里可以设置更新的频率)
    end for
end for
"""


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


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(QNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(
                nn.ModuleDict({
                    'linear': nn.Linear(state_dim if not idx else hidden_layers_dim[idx-1], h),
                    'linear_active': nn.ReLU(inplace=True)
                    
                })
            )
        self.header = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.header(x)

    def model_compelet(self, learning_rate):
        self.cost_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()
        

class DQN:
    def __init__(self, 
                state_dim: int, 
                hidden_layers_dim, 
                action_dim: int, 
                learning_rate: float,
                gamma: float,
                epsilon: float=0.05,
                traget_update_freq: int=1,
                device: typ.AnyStr='cpu',
                dqn_type: typ.AnyStr='DQN'
                ):
        self.action_dim = action_dim
        # QNet & targetQNet
        self.q = QNet(state_dim, hidden_layers_dim, action_dim)
        self.target_q = copy.deepcopy(self.q)
        self.q.to(device)
        self.q.model_compelet(learning_rate)
        self.target_q.to(device)

        # iteration params
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # target update freq
        self.traget_update_freq = traget_update_freq
        self.count = 0
        self.device = device
        
        # dqn类型
        self.dqn_type = dqn_type
        
    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        action = self.target_q(torch.FloatTensor(state))
        return np.argmax(action.detach().numpy())
    
    def update(self, samples: deque):
        """
        Q<s, a, t> = R<s, a, t> + gamma * Q<s+1, a_max, t+1>
        """
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)
        
        states = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).view(-1, 1).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        actions_q = self.q(states)
        n_actions_q = self.target_q(next_states)
        q_values = actions_q.gather(1, action)
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = n_actions_q.gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = n_actions_q.max(1)[0].view(-1, 1)

        q_targets = reward + self.gamma * max_next_q_values * (1 - done) 
        # MSELoss update
        self.q.update(q_values.float(), q_targets.float())
        if self.count % self.traget_update_freq == 0:
            self.target_q.load_state_dict(
                self.q.state_dict()
            )


def Pendulum_dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    action_range = action_upbound - action_lowbound
    return action_lowbound + (discrete_action / (action_dim - 1)) * action_range


def play(env, env_agent, cfg, episode_count=2, action_contiguous=False):
    """
    对训练完成的QNet进行策略游戏
    """
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            env.render()
            a = env_agent.policy(s)
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_state, reward, done, _, _ = env.step([c_a])
            else:
                n_state, reward, done, _, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (episode_reward >= 3 * cfg.max_episode_rewards) or (episode_cnt >= 3 * cfg.max_episode_steps):
                break

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')

    env.close()

class Config:
    num_episode  = 300
    state_dim = None
    hidden_layers_dim = [10, 10]
    action_dim = 20
    learning_rate = 2e-3
    gamma = 0.95
    epsilon = 0.01
    traget_update_freq = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 2048
    minimal_size = 1024
    batch_size = 128
    render = False
    save_path =  r'D:\TMP\model.ckpt' 
    dqn_type = 'DoubleDQN'
    # 回合停止控制
    max_episode_rewards = 260
    max_episode_steps = 260

    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            pass
        print(f'device = {self.device} | env={str(env)}')
    


def train_dqn(env, cfg, action_contiguous=False):
    buffer = replayBuffer(cfg.buffer_size)
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim, 
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        traget_update_freq=cfg.traget_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    for i in tq_bar:
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = dqn.policy(s)
            # [Any, float, bool, bool, dict]
            if action_contiguous:
                c_a = Pendulum_dis_to_con(a, env, cfg.action_dim)
                n_s, r, done, _, _ = env.step([c_a])
            else:
                n_s, r, done, _, _ = env.step(a)
            buffer.add(s, a, r, n_s, done)
            s = n_s
            episode_rewards += r
            steps += 1
            # buffer update
            if len(buffer) > cfg.minimal_size:
                samples = buffer.sample(cfg.batch_size)
                dqn.update(samples)
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break

        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if bf_reward < now_reward:
            torch.save(dqn.target_q.state_dict(), cfg.save_path)

        bf_reward = max(bf_reward, now_reward)
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    return dqn


if __name__ == '__main__':
    # print('=='*35)
    # print('Training CartPole-v0')
    # env = gym.make('CartPole-v0') #, render_mode="human")
    # cfg = Config(env)
    # dqn = train_dqn(env, cfg)
    # dqn.target_q.load_state_dict(torch.load(cfg.save_path))
    # play(gym.make('CartPole-v0', render_mode="human"), dqn, cfg)
    
    print('=='*35)
    print('Training Pendulum-v1')
    p_env = gym.make('Pendulum-v1')
    p_cfg = Config(p_env)
    p_dqn = train_dqn(p_env, p_cfg, True)
    p_dqn.target_q.load_state_dict(torch.load(p_cfg.save_path))
    play(gym.make('Pendulum-v1', render_mode="human"), p_dqn, p_cfg, episode_count=2, action_contiguous=True)







