# python3
# Author: Scc_hy
# Create Date: 2025-05-26
# Func: A2C
# reference: https://chihhuiho.github.io/project/ucsd_ece_276/report.pdf
#            OpenAI Baselines: ACKTR & A2C https://openai.com/index/openai-baselines-acktr-a2c/
#            Sample Efficient Actor-Critic with Experience Replay https://arxiv.org/abs/1611.01224 
#            baselines3 https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py 
#            https://github.com/younggyoseo/pytorch-acer/blob/master/algo/acer.py
# ============================================================================
import typing as typ
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from ._base_net import PPOValueNet as valueNet
from ._base_net import SACPolicyNet as policyNet


class A2CER:
    """ 
    
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
    """
    def __init__(
        self,
        state_dim: int,
        actor_hidden_layers_dim: typ.List[int],
        critic_hidden_layers_dim: typ.List[int],
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        A2C_kwargs: typ.Dict,
        device: torch.device   
    ):
        self.state_dim = state_dim
        self.actor_hidden_layers_dim = actor_hidden_layers_dim
        self.critic_hidden_layers_dim = critic_hidden_layers_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device
        self.value_net = valueNet(state_dim, critic_hidden_layers_dim, A2C_kwargs.get('act_type', 'relu')).to(device)
        self.v_opt = optim.RMSprop(self.value_net.parameters(), lr=critic_lr)
        self.v_cost_func = nn.MSELoss()
        
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, A3C_kwargs.get('action_bound', 1.0)).to(device)
        self.a_opt = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
    
    def policy(self, state):
        a, log_prob = self.actor(state)
        return a

    def update(self, samples_buffer: deque, wandb = None, update_lr=True):
        state, action, reward, next_state, done, adv = zip(*samples)
        adv = torch.FloatTensor(np.stack(adv)).to(self.device)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = np.stack(reward).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = np.stack(done).to(self.device)

        v_t = self.value_net(state)
        v_tar = adv + v_t
        # value net loss 
        self.v_opt.zero_grad()
        loss = self.v_cost_func(v_tar.float(), v_t.float())
        loss.backward()
        self.v_opt.step()
        
        # \partial log(\pi(a|s, \theta))}{\partial \theta} A
        a, log_prob = self.actor(state)
        actor_loss = -(log_prob * adv).mean()
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()


