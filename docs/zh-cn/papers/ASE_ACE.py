import numpy as np
import gymnasium as gym 
import math
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt

print(np.__version__) # 1.24.4
print(gym.__version__) # 0.29.1


class ASE:
    def __init__(self, input_dim, alpha=0.1, delta=0.2, sigma=0.01):
        """associative search element"""
        self.w = np.random.randn(input_dim, 1)
        self.lr = alpha
        self.et = 0
        self.delta = delta
        self.sigma = sigma

    def forward(self, state):
        noise = np.random.normal(0, self.sigma)
        return  self.f_func(np.dot(state, self.w) + noise)
    
    def f_func(self, a):
        if a < 0:
            return 0
        return 1
    
    def action_postfix(self, a):
        if a == 0:
            return -1
        return 1      

    def update(self, s, hat_r):
        # 1- conditions of action: becomes eligible to have its weight modified
        e_t = self.action_postfix(self.forward(s)) * s
        # w^i_{t+1} = w^i_t + \alpha r_t e^i_t
        self.w += (self.lr * hat_r * e_t).reshape(self.w.shape)
        # 2- it remains eligible for some period of time after the conditions cease to hold
        # e^i_{t+1} = \delta e^i_{t} + (1-\delta)y_t^ix_t^i
        self.et = self.delta * self.et + (1 - self.delta) * e_t


class ACE:
    """adaptive critic element"""
    def __init__(self, input_dim, beta=0.1, gamma=0.95, lmbda=0.8):
        self.v = np.random.randn(input_dim)
        self.lr = beta
        self.gamma = gamma
        self.lmbda = lmbda
        self.overline_x = np.zeros(input_dim)
    
    def forward(self, state):
        return np.dot(state, self.v) 
    
    def reward_postfix(self, r):
        return r - 1
    
    def reinforcement_signal(self, r, s, before_s):
        r = self.reward_postfix(r) # 0 -1
        return r + self.gamma * self.forward(s) - self.forward(before_s)

    def update(self, s, r, before_s):
        # r_t + \gamma p_t - p_{t-1}
        hat_r = self.reinforcement_signal(r, s, before_s)
        
        # v^i_{t+1} = v^i_t + \beta \overline{x}_i
        self.v += self.lr * hat_r * self.overline_x 

        # \overline{x}^i_{t+1} = \lambda \overline{x}^i_t + ( 1 - \lambda) x^i_t
        self.overline_x = self.lmbda * self.overline_x + (1 - self.lmbda) * s
        return hat_r


# 创建环境
seed_ = 19831983
np.random.seed(seed_)
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

 
# paper 1000
ase = ASE(input_dim, alpha=580, delta=0.9, sigma=0.01) 
ace = ACE(input_dim, beta=0.5, gamma=0.95, lmbda=0.8)

num_episodes = 1200 # 500000
reward_l = []
hat_r_list = []
tq_bar = tqdm(range(num_episodes))
for episode in tq_bar:
    s, _ = env.reset(seed=20250314)   # 500
    done = False
    r_tt = 0
    hat_r_ep_list = []
    while not done:
        a = ase.forward(s)  
        n_s, r, terminated, truncated, infos = env.step(a) 
        r_tt += r
        # 更新 ACE
        hat_r = ace.update(n_s, r, s)
        hat_r_ep_list.append(hat_r)
        # 更新 ASE
        ase.update(n_s, hat_r)
        s = n_s
        done = terminated or truncated

    reward_l.append(r_tt)
    hat_r_list.append(np.mean(hat_r_ep_list))
    tq_bar.set_postfix({
        'rewards': r_tt,
        'last_mean': np.mean(reward_l[-10:]),
        'last_std': np.std(reward_l[-10:]),
        'hat_r_mean': np.mean(hat_r_list[-10:]),
    })

plt.plot(range(len(hat_r_list[750:])), hat_r_list[750:])
plt.title('hat_r\nepsiode 750~1200')
plt.savefig('hat_r.png')


env = gym.make('CartPole-v1', render_mode='human')
num_test = 3
tq_bar = tqdm(range(num_test))
for episode in tq_bar:
    s, _ = env.reset(seed=20250314)
    done = False
    r_tt = 0
    while not done:
        env.render()
        a = ase.forward(s)  
        n_s, r, terminated, truncated, infos = env.step(a) 
        r_tt += r
        s = n_s
        done = terminated or truncated


    reward_l.append(r_tt)
    tq_bar.set_postfix({
        'rewards': r_tt,
        'last_mean': np.mean(reward_l[-10:]),
        'last_std': np.std(reward_l[-10:]),
    })
    
env.close()
