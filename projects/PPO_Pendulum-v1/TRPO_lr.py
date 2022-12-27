# python3
# Create Dat3: 2022-12-26
# Func: TPRO(Trust Region Policy Optimization)
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
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(policyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , action_dim)
        self.action_dim = action_dim

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        x = self.head(x)
        return F.softmax(x - torch.max(x), dim=1)




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
    advantage_list = []
    adv = 0
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        advantage_list.append(adv)
    advantage_list.reverse()
    return torch.FloatTensor(advantage_list)



class TRPO:
    def __init__(self, 
                state_dim,
                hidden_layers_dim,
                action_dim,
                actor_lr,
                critic_lr,
                gamma,
                TRPO_kwargs,
                device
                ):
        self.state_dim = state_dim
        self.actor = policyNet(state_dim, hidden_layers_dim, action_dim).to(device)
        self.critic = valueNet(state_dim, hidden_layers_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = TRPO_kwargs['lmbda']  # GAE参数
        self.kl_constraint = TRPO_kwargs['kl_constraint']   # KL距离最大限制
        self.alpha = TRPO_kwargs['alpha'] # 线性搜索参数
        self.device = device
        self.count = 0
    
    def policy(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算hession matix和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
        )
        # 计算梯度
        kl_grad = torch.autograd.grad(
            kl,
            self.actor.parameters(),
            create_graph=True
        )
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product,
            self.actor.parameters()
        )
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = r @ r
        for _ in range(10): # 共轭主循环
            Hp = self.hessian_matrix_vector_product(
                states, old_action_dists, p
            )
            
            alpha  = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = r @ r
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        log_probs = torch.log(actor(states).gather(1, actions))   
        ratio = torch.exp(log_probs - old_log_probs) 
        return torch.mean(ratio * advantage)
    
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):
        # 计算策略目标
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        for i in range(5):
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )
            new_action_dists = torch.distributions.Categorical(
                new_actor(states)
            )
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists, new_action_dists
                )
            )
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(
            obj_grad, states, old_action_dists
        )
        Hd = self.hessian_matrix_vector_product(
            states, old_action_dists, descent_direction
        )
        max_coef = torch.sqrt(2 * self.kl_constraint / ( descent_direction @ Hd + 1e-8))
        new_para = self.line_search(
            states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef
        )
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略

    def update(self, samples):
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).view(-1, 1).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
       
        td_target = reward + self.gamma * self.critic(next_state) * ( 1 - done)
        td_dela = td_target - self.critic(state)
        advantage = compute_advantage(self.gamma, self.lmbda, td_dela.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(state).gather(1, action)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(state).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        # 更新策略函数
        self.policy_learn(state, action, old_action_dists, old_log_probs, advantage)


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
    num_episode = 500
    state_dim = None
    hidden_layers_dim = [ 128, 128 ]
    action_dim = 20
    actor_lr = 1e-3
    critic_lr = 1e-2
    TRPO_kwargs = {
        'lmbda': 0.9,
        'alpha': 0.5,
        'kl_constraint': 0.00005
    }
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    buffer_size = 20480
    minimal_size = 1024
    batch_size = 128
    save_path = r'D:\tmp\ac_model.ckpt'
    # 回合停止控制
    max_episode_rewards = 260
    max_episode_steps = 260
    
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            pass
        print(f'device={self.device} | env={str(env)}')


def train_agent(env, cfg):
    ac_agent = TRPO(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        TRPO_kwargs=cfg.TRPO_kwargs,
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
    print('Training CartPole-v0')
    env = gym.make('CartPole-v0')
    cfg = Config(env)
    ac_agent = train_agent(env, cfg)
    ac_agent.actor.load_state_dict(torch.load(cfg.save_path))
    play(gym.make('CartPole-v0', render_mode="human"), ac_agent, cfg)

