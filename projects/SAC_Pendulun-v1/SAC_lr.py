# python3
# Create Date 2023-01-18
# Func: SAC
# ========================================================================================


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import typing as typ
from torch.distributions import Normal
from copy import deepcopy
from collections import deque
import random
from tqdm import tqdm
import gym




class policyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int, action_bound: float=1.0):
        super(policyNet, self).__init__()
        self.action_bound = action_bound
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
        
        mean_ = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 3e-5
        dist = Normal(mean_, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return  action * self.action_bound, log_prob




class valueNet(nn.Module):
    def __init__(self, state_action_dim: int, hidden_layers_dim: typ.List):
        super(valueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_action_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1).float() # 拼接状态和动作
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x) 






class SAC:
    """
    一个策略网络
    两个价值网络
    两个目标网络
    """
    def __init__(
        self,
        state_dim: int,
        hidden_layers_dim: typ.List[int],
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        SAC_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.actor = policyNet(state_dim, hidden_layers_dim, action_dim, action_bound=SAC_kwargs.get('action_bound', 1.0)).to(device)
        self.critic_1 = valueNet(state_dim+action_dim, hidden_layers_dim)
        self.critic_2 = valueNet(state_dim+action_dim, hidden_layers_dim)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)
        self._critic_to_device(device)
        
        # optim
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.gamma = gamma
        self.device = device
        self.tau = SAC_kwargs.get('tau', 0.05)
        self.target_entropy = SAC_kwargs['target_entropy']
            
    def _critic_to_device(self, device):
        self.critic_1 = self.critic_1.to(device)
        self.critic_2 = self.critic_2.to(device)
        self.target_critic_1 = self.target_critic_1.to(device)
        self.target_critic_2 = self.target_critic_2.to(device)
    
    def policy(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = self.actor(state)[0]
        return action.detach().numpy()[0]


    def calc_target(self, reward, next_state, dones):
        # 计算目标Q值
        next_actions, log_prob = self.actor(next_state)
        # entropy = -log_prob
        q1_v = self.target_critic_1(next_state, next_actions)
        q2_v = self.target_critic_2(next_state, next_actions)
        next_v = torch.min(q1_v, q2_v) - self.log_alpha.exp() * log_prob
        return reward + self.gamma * next_v * ( 1 - dones )


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )


    def update(self, samples):
        state, action, reward, next_state, done = zip(*samples)


        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        reward = (reward + 10.0) / 10.0  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        td_target = self.calc_target(reward, next_state, done)
        # update critic net
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(state, action).float(), td_target.float().detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(state, action).float(), td_target.float().detach())
        )
        self.critic_1_opt.zero_grad()
        critic_1_loss.backward()
        self.critic_1_opt.step()
        self.critic_2_opt.zero_grad()
        critic_2_loss.backward()
        self.critic_2_opt.step()
        
        # update actor
        new_act, log_prob = self.actor(state)
        q1_v = self.target_critic_1(next_state, new_act)
        q2_v = self.target_critic_2(next_state, new_act)
        actor_loss = torch.mean(self.log_alpha.exp() * log_prob - torch.min(q1_v, q2_v))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
    
        # update alpha
        alpha_loss = torch.mean((-log_prob - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


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
    num_episode = 230
    state_dim = None
    hidden_layers_dim = [ 64, 64 ]
    action_dim = 20
    actor_lr = 3e-5
    critic_lr = 5e-4
    alpha_lr = 5e-4
    SAC_kwargs = {
        'tau': 0.05, # soft update parameters
        'target_entropy': 0.01,
        'action_bound': 1.0
    }
    gamma = 0.95
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 10240
    minimal_size = 1024
    batch_size = 128
    save_path = r'D:\tmp\SAC_ac_model.ckpt'
    # 回合停止控制
    max_episode_rewards = 204800
    max_episode_steps = 260


    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
            self.SAC_kwargs['target_entropy'] = -env.action_space.shape[0]
        except Exception as e:
            self.action_dim = env.action_space.shape[0]
            self.SAC_kwargs['target_entropy'] = -env.action_space.shape[0]
        print(f'device={self.device} | env={str(env)}')


def train_agent(env, cfg):
    ac_agent = SAC(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = []
    now_reward = 0
    bf_reward = -np.inf
    buffer_ = replayBuffer(cfg.buffer_size)
    for i in tq_bar:
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
            # buffer update
            if len(buffer_) > cfg.minimal_size:
                samples = buffer_.sample(cfg.batch_size)
                ac_agent.update(samples)
            if (episode_rewards >= cfg.max_episode_rewards) or (steps >= cfg.max_episode_steps):
                break


        rewards_list.append(episode_rewards)
        now_reward = np.mean(rewards_list[-10:])
        if (bf_reward < now_reward) and (i >= 10):
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
    # ac_agent = SAC(
    #     state_dim=cfg.state_dim,
    #     hidden_layers_dim=cfg.hidden_layers_dim,
    #     action_dim=cfg.action_dim,
    #     actor_lr=cfg.actor_lr,
    #     critic_lr=cfg.critic_lr,
    #     alpha_lr=cfg.alpha_lr,
    #     gamma=cfg.gamma,
    #     SAC_kwargs=cfg.SAC_kwargs,
    #     device=cfg.device
    # )
    ac_agent.actor.load_state_dict(torch.load(cfg.save_path))
    play(gym.make('Pendulum-v1', render_mode="human"), ac_agent, cfg)

