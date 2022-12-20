# python3
# Create Date:2022-12-20
# Author: Scc_hy
# Func:  PolictNet
# Tip: 
#     replayBuffer 一个经验回放池
#     policyNet 策略网络
#     REINFORCE: 蒙特卡洛策略梯度
#     Config: 训练的时候的重要参数
# =================================================================================

from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import typing as typ
import copy
import gym
from tqdm import tqdm
import os

print(gym.__version__) # 0.26.2

class replayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)


class policyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(policyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(
                nn.ModuleDict({
                    'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                    'linear_activation': nn.ReLU(inplace=True)
                })
            )
        self.header = nn.Linear(hidden_layers_dim[-1], action_dim)
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](
                layer['linear'](x)
            )
        return F.softmax(self.header(x))
    
    def model_compelete(self, learning_rate):
        self.cost_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def update(self, pred, target):
        self.opt.zero_grad()
        loss = self.cost_func(pred, target)
        loss.backward()
        self.opt.step()



class REINFORCE:
    def __init__(self,
                state_dim,
                hidden_layers_dim,
                action_dim,
                learning_rate,
                gamma,
                device):
        self.policy_net = policyNet(
            state_dim=state_dim,
            hidden_layers_dim=hidden_layers_dim,
            action_dim=action_dim
        ).to(device)
        self.policy_net.model_compelete(learning_rate)
        self.gamma = gamma
        self.device = device
        self.count = 0
    
    def policy(self, state):
        s = torch.FloatTensor(state).to(self.device)
        probs = self.policy_net(s)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, samples: deque):
        self.count += 1  
        state, action, reward, next_state, done = zip(*samples)
        G = 0
        self.policy_net.opt.zero_grad()
        for i in range(len(reward)-1, 0, -1):
            r = reward[i]
            s = torch.FloatTensor([state[i]]).to(self.device)
            a = torch.tensor([action[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(s).gather(1, a))
            G = r + self.gamma * G
            loss = -log_prob * G
            loss.backward()
        self.policy_net.opt.step()

        

def play(env, env_agent, cfg, episode_count=2):
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
            n_state, reward, done, _, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (episode_reward >= 3 * cfg.max_episode_rewards) or (episode_cnt >= 3 * cfg.max_episode_steps):
                break

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')

    env.close()
            
            

class Config:
    num_episode  = 1000
    state_dim = None
    hidden_layers_dim = [20, 20]
    action_dim = 20
    learning_rate = 2e-3
    gamma = 0.95
    traget_update_freq = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 20480
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
    


def train_agent(env, cfg):
    rf = REINFORCE(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        device=cfg.device
    )
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    for i in tq_bar:
        buffer = replayBuffer(cfg.buffer_size)
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = rf.policy(s)
            n_s, r, done, _, _ = env.step(a)
            buffer.add(s, a, r, n_s, done)
            s = n_s
            episode_rewards += r
            steps += 1
            # buffer update
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break
        
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        rf.update(buffer.buffer)
        if bf_reward < now_reward:
            torch.save(rf.policy_net.state_dict(), cfg.save_path)
            bf_reward = now_reward

        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    return rf


if __name__ == '__main__':
    print('=='*35)
    print('Training CartPole-v0')
    env = gym.make('CartPole-v0') #, render_mode="human")
    cfg = Config(env)
    rf = train_agent(env, cfg)
    rf.policy_net.load_state_dict(torch.load(cfg.save_path))
    play(gym.make('CartPole-v0', render_mode="human"), rf, cfg)
