# python3
# Create Date:2022-12-17
# Author: Scc_hy
# Func:  Dueling DQN
# Tip: 
#     replayBuffer 一个经验回放池
#     VANet: 一个网络，顶层分别输出action 和 value
#     DQN:  dueling dqn整体框架
#     Config: 训练的时候的重要参数
# =================================================================================


import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from torch import optim
import gym
import typing as typ
import copy
from collections import deque
import random
from tqdm import tqdm

print(gym.__version__)


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



class VANet(nn.Module):
    # 可以让智能体开始关注不同动作优势值的差异
    # Dueling DQN 能够更加频繁、准确地学习状态价值函数
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(VANet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            })) 
        
        # 表示采取不同动作的差异性
        self.adv_header = nn.Linear(hidden_layers_dim[-1], action_dim)
        # 状态价值函数
        self.v_header = nn.Linear(hidden_layers_dim[-1], 1)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        
        adv = self.adv_header(x)
        v = self.v_header(x)
        # Q值由V值和A值计算得到
        Q = v + adv - adv.mean().view(-1, 1)  
        return Q

    def complete(self, lr):
        self.cost_func = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)
    
    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()



class QNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(QNet, self).__init__()
        self.featires = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_active': nn.ReLU(inplace=True)
            }))
        self.head = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)

    def complete(self, lr):
        self.cost_func = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)
    
    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()



class DQN:
    def __init__(self, 
                state_dim: int,
                hidden_layers_dim: typ.List[int],
                action_dim: int,
                learning_rate: float,
                gamma: float,
                epsilon: float=0.05,
                target_update_freq: int=1,
                device: typ.AnyStr='cpu',
                dqn_type: typ.AnyStr='DQN'
                ):
        
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':
            self.q = VANet(state_dim, hidden_layers_dim, action_dim).to(device)
        else:
            self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(device)

        self.target_q = copy.deepcopy(self.q)
        self.q.complete(lr=learning_rate)
        self.target_q.to(device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device
        self.dqn_type = dqn_type
        self.count = 0

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        action = self.target_q(torch.FloatTensor(state))
        return np.argmax(action.detach().numpy())

    def update(self, samples: deque):
        self.count += 1
        # sample -> tensor
        states, actions, rewards, next_states, done = zip(*samples)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.Tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.Tensor(rewards).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.Tensor(done).to(self.device)
        
        Q_s = self.q(states)
        Q1_s = self.target_q(next_states)
        # print(Q_s.shape, actions.shape)
        Q_sa = Q_s.gather(1, actions.long())
        # 下一状态最大值
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q(next_states).max(1)[1].view(-1, 1)
            max_Q1sa = Q1_s.gather(1, max_action)
        else:
            max_Q1sa = Q1_s.max(1)[0].view(-1, 1)

        Q_target = rewards + self.gamma * max_Q1sa * (1 - done)
        # MSELoss
        self.q.update(Q_sa.float(), Q_target.float())
        if self.count % self.target_update_freq == 0:
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
    num_episode  = 100
    state_dim = None
    hidden_layers_dim = [10, 10]
    action_dim = 20
    learning_rate = 2e-3
    gamma = 0.95
    epsilon = 0.01
    target_update_freq = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 2048
    minimal_size = 1024
    batch_size = 128
    render = False
    save_path =  r'D:\TMP\new_model.ckpt' 
    dqn_type = 'DuelingDQN'
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
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = -np.inf
    bf_reward = -np.inf
    for i in tq_bar:
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = dqn.policy(s)
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
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_rewards):
                break
            
            rewards_list.append(episode_rewards)
            now_reward = np.mean(rewards_list[-10:])
            if bf_reward < now_reward:
                torch.save(dqn.target_q.state_dict(), cfg.save_path)
                bf_reward = now_reward
                
            tq_bar.set_postfix({
                'lastMeanRewards': f'{now_reward:.2f}',
                'BEST': f'{bf_reward:.2f}'
            })
    env.close()
    return dqn
            
            

if __name__ == '__main__':
    print('=='*35)
    env_name = 'Pendulum-v1'
    print(f'Training {env_name}')
    env = gym.make(env_name)
    cfg = Config(env)
    dqn = train_dqn(env, cfg, True)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2, action_contiguous=True)


