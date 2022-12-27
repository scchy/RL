# python3
# Create Dat3: 2022-12-27
# Func: PPO 输出action为连续变量
# =====================================================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import gym
import copy
import random
from collections import deque
from tqdm import tqdm
import typing as typ



class policyNet(nn.Module):
    """
    continuity action:
    normal distribution (mean, std) 
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(policyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))

        self.fc_mu = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layers_dim[-1], action_dim)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        
        mean_ = 2.0 * torch.tanh(self.fc_mu(x))
        # np.log(1 + np.exp(2))
        std = F.softplus(self.fc_std(x))
        return mean_, std


class valueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim):
        super(valueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)
        
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    adv_list = []
    adv = 0
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        adv_list.append(adv)
    adv_list.reverse()
    return torch.FloatTensor(adv_list)


class PPO:
    """
    PPO算法, 采用截断方式
    """
    def __init__(self,
                state_dim: int,
                hidden_layers_dim: typ.List,
                action_dim: int,
                actor_lr: float,
                critic_lr: float,
                gamma: float,
                PPO_kwargs: typ.Dict,
                device: torch.device
                ):
        self.actor = policyNet(state_dim, hidden_layers_dim, action_dim).to(device)
        self.critic = valueNet(state_dim, hidden_layers_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.ppo_epochs = PPO_kwargs['ppo_epochs'] # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps'] # PPO中截断范围的参数
        self.count = 0 
        self.device = device
    
    def policy(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
        
    def update(self, samples: deque):
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).view(-1, 1).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        reward = (reward + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_delta = td_target - self.critic(state)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
                
        mu, std = self.actor(state)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(action)
        for _ in range(self.ppo_epochs):
            mu, std = self.actor(state)
            action_dists = torch.distributions.Normal(mu, std)
            log_prob = action_dists.log_prob(action)
            
            # e(log(a/b))
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2)).float()
            critic_loss = torch.mean(
                F.mse_loss(self.critic(state).float(), td_target.detach().float())
            ).float()
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()




class replayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state, done) )
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
        

def play(env, env_agent, cfg, episode_count=2):
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
            if (episode_cnt >= 3 * cfg.max_episode_steps) or (episode_reward >= 3*cfg.max_episode_rewards):
                break
    
        print(f'Get reward {episode_reward}. Last {episode_cnt} times')
    env.close()








class Config:
    num_episode = 1200
    state_dim = None
    hidden_layers_dim = [ 128, 128 ]
    action_dim = 20
    actor_lr = 5e-4
    critic_lr = 5e-3
    PPO_kwargs = {
        'lmbda': 0.95,
        'eps': 0.2,
        'ppo_epochs': 10
    }        
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 20480
    minimal_size = 1024
    batch_size = 128
    save_path = r'D:\TMP\ac_model.ckpt'
    # 回合停止控制
    max_episode_rewards = 260
    max_episode_steps = 260
    
    
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            self.action_dim = env.action_space.shape[0]
        print(f'device={self.device} | env={str(env)}')



def train_agent(env, cfg):
    ac_agent = PPO(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device
    )           
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    for i in tq_bar:
        buffer_ = replayBuffer(cfg.buffer_size)
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ]')    
        s, _ = env.reset()
        done = False
        episode_rewards = 0
        steps = 0
        while not done:
            a = ac_agent.policy(s)
            n_s, r, done, _, _ = env.step(a)
            buffer_.add(s, a, r, n_s, done)
            s = n_s
            episode_rewards += r
            steps += 1
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break

        ac_agent.update(buffer_.buffer)
        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if bf_reward < now_reward:
            torch.save(ac_agent.actor.state_dict(), cfg.save_path)
            bf_reward = now_reward
        
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    return ac_agent






if __name__ == '__main__':
    print('=='*35)
    print('Training Pendulum-v1')
    env = gym.make('Pendulum-v1')
    cfg = Config(env)
    ac_agent = train_agent(env, cfg)
    ac_agent.actor.load_state_dict(torch.load(cfg.save_path))
    play(gym.make('Pendulum-v1', render_mode="human"), ac_agent, cfg)

