# python3
# Author: Scc_hy
# Create Date: 2025-05-26
# Func: A2C
# reference: Main Paper https://chihhuiho.github.io/project/ucsd_ece_276/report.pdf
#            OpenAI Baselines: ACKTR & A2C https://openai.com/index/openai-baselines-acktr-a2c/
#            Sample Efficient Actor-Critic with Experience Replay https://arxiv.org/abs/1611.01224 
#            baselines3 https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py 
#            https://github.com/younggyoseo/pytorch-acer/blob/master/algo/acer.py
# ============================================================================
import typing as typ
import numpy as np
import torch 
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from ._base_net import ActorCriticorMLP

 
class memDataset(Dataset):
    def __init__(self, states: tensor, actions: tensor, old_log_probs: tensor, ut: tensor, done: tensor):
        super(memDataset, self).__init__()
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.ut = ut
        self.done = done

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        old_log_probs = self.old_log_probs[index]
        ut = self.ut[index]
        done = self.done[index]
        return states, actions, old_log_probs, ut, done
    

class SEA2CER:
    """ 
    Sample Efficient Actor-Critic with Experience Replay : adding experience replay to original A2C algrithm
    
    PaperIdea: A3C + ACER (Actor-Critic with Expirence Reply); on-policy -> off-policy
    problem: 
        - solve the instability when training the network in the same thread
        - Expirence Reply: 缓解问题, 但计算和内存资源的占用高。 
    solve:
        - 引入多个Agent在不同线程上并行与环境交互来解决这个问题
        - 每个Agen都是随机初始化的, 因此不同代理的经验之间的数据相关性降低了
        - 方法[8]允许在不占用大量计算资源的情况下进行策略学习，并提高了训练过程中的稳定性。
        - reward: applied the softmax to the reward function when we need to prioritize the replay buffer
    
    actor:  \pi(a|s, \theta)
    critic-value_net: V(s)
    policyLoss = \frac{\partial log(\pi(a|s, \theta))}{\partial \theta} A
    Advantage: A = r+\gamma V(s_{t+1}) - V(s_t)
    
    A3C: 3个核idea
    1. operates on fixed-length segments of experience (say, 20 timesteps) and uses these segments to compute estimators of the returns and advantage function.
    2. architectures that 【share layers】 between the policy and value function. 
    3. the networks are updated asynchronously.
    
    Sample Efficient Actor-Critic with Experience Replay https://arxiv.org/abs/1611.01224 
    $Q^{ret}(x_t, a_t) = r_t + \gamma \hat{\rho}_{t+1}[Q^{ret}(x_{t+1}, a_{t+1}) - Q(x_{t+1}, a_{t+1})] + \gamma V(x_{t+1})$
    - consider only \lambda = 1
    - $\hat{\rho}_{t}=min(c, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)})$
    Critic loss: $\nabla _\theta L = (Q(x_t, a_t) - Q^{ret}(x_t, a_t))\nabla _\theta Q(x_t, a_t)$
    
    
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers_dim: typ.List[int],
        actor_hidden_layers_dim: typ.List[int],
        critic_hidden_layers_dim: typ.List[int],
        learning_rate: float,
        gamma: float,
        A2C_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.state_dim = state_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.actor_hidden_layers_dim = actor_hidden_layers_dim
        self.critic_hidden_layers_dim = critic_hidden_layers_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        self.clip = A2C_kwargs.get('clip', 0.5)
        self.entroy_coef = A2C_kwargs.get('entroy_coef', None)
        self.entroy_org = self.entroy_coef
        self.critic_coef = A2C_kwargs.get('critic_coef', 1.0)
        self.max_grad_norm = A2C_kwargs.get('max_grad_norm', 0)
        self.k_epochs = A2C_kwargs.get('k_epochs', 3)
        self.normalize_adv = A2C_kwargs.get('normalize_adv', False)
        self.minibatch_size = A2C_kwargs.get('minibatch_size', 256)
        self.learning_rate = learning_rate
        self.actor_critic = ActorCriticorMLP(
            action_dim, 
            state_dim, 
            hidden_layers_dim, 
            critic_hidden_layers_dim, 
            actor_hidden_layers_dim, 
            act_type=A2C_kwargs.get('act_type', 'relu'), 
            action_type=A2C_kwargs.get('env_action_type', 'Cat')
        ).to(device)
        self.ac_opt = optim.RMSprop(self.actor_critic.parameters(), lr=learning_rate)
    
    def train(self):
        self.actor_critic.train()

    def eval(self):
        self.actor_critic.eval()
    
    def update_entroy_coef(self, ep, num_episode):
        self.entroy_coef = max(self.entroy_org * (1 - ep/num_episode/10), 0.01)
    
    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        ac_dist = self.actor_critic(state, 'get_dist')
        a = ac_dist.sample()
        return a.detach().cpu().numpy()[0], ac_dist.log_prob(a).detach().cpu().numpy()

    def update(self, samples_buffer, wandb = None, update_lr=True):
        state, action, reward, next_state, done, actor_log_prob, R = zip(*samples_buffer)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = torch.LongTensor(np.stack(reward)).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        actor_log_prob = torch.FloatTensor(np.stack(actor_log_prob)).to(self.device)
        done = np.stack(done) # .to(self.device)
        R = torch.LongTensor(np.stack(R)).to(self.device)
        d_set = memDataset(state, action, actor_log_prob, R, torch.Tensor(done).to(self.device))
        train_loader = DataLoader(
            d_set,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True
        )

        for mini_ep in range(self.k_epochs):
            # 梯度累积
            self.ac_opt.zero_grad()
            for state_, action_, old_log_prob, R_, d_ in train_loader:
                new_dist, new_v = self.actor_critic(state_)
                entropy_loss = new_dist.entropy().mean()
                new_log_prob = new_dist.log_prob(action_)
                try:
                    new_log_prob = new_log_prob.sum(1)
                    old_log_prob = old_log_prob.sum(1).sum(1)
                except Exception as e:
                    pass
                imp = torch.exp(new_log_prob - old_log_prob.detach())

                adv = (R_ - new_v.view(-1)) * (1 - d_)
                
                critic_loss = 0.5 * (adv ** 2).mean() 
                if self.normalize_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8) 

                actor_loss = -((imp * adv).detach() * new_log_prob).mean()
                if self.entroy_coef is not None:
                    actor_loss -= self.entroy_coef * entropy_loss
                total_loss = actor_loss + self.critic_coef * critic_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) 
            
            self.ac_opt.step()

    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        ac_f = os.path.join(file_path, 'SEA2CER_ac.ckpt')
        torch.save(self.actor_critic.state_dict(), ac_f)

    def load_model(self, file_path):
        ac_f = os.path.join(file_path, 'SEA2CER_ac.ckpt')
        try:
            self.actor_critic.load_state_dict(torch.load(ac_f))
        except Exception as e:
            self.actor_critic.load_state_dict(torch.load(ac_f, map_location='cpu'))

        self.actor_critic.to(self.device)
        self.ac_opt = optim.RMSprop(self.actor_critic.parameters(), lr=self.learning_rate)
    
