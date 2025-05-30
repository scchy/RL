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
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from ._base_net import QNet  
from ._base_net import PPOValueNet as valueNet
from ._base_net import PPOPolicyBetaNet as policyNet

 
class memDataset(Dataset):
    def __init__(self, states: tensor, actions: tensor, old_log_probs: tensor, ut: tensor):
        super(memDataset, self).__init__()
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.ut = ut

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        old_log_probs = self.old_log_probs[index]
        ut = self.ut[index]
        return states, actions, old_log_probs, ut 
    

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
    
    Sample Efficient Actor-Critic with Experience Replay https://arxiv.org/abs/1611.01224 
    $Q^{ret}(x_t, a_t) = r_t + \gamma \hat{\rho}_{t+1}[Q^{ret}(x_{t+1}, a_{t+1}) - Q(x_{t+1}, a_{t+1})] + \gamma V(x_{t+1})$
    - consider only \lambda = 1
    - $\hat{\rho}_{t}=min(c, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)})$
    Critic loss: $\nabla _\theta L = (Q(x_t, a_t) - Q^{ret}(x_t, a_t))\nabla _\theta Q(x_t, a_t)$
    
    
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
        self.clip = A2C_kwargs.get('clip', 0.5)
        self.k_epochs = A2C_kwargs.get('k_epochs', 3)
        self.minibatch_size = A2C_kwargs.get('minibatch_size', 256)
        self.q_net = QNet(state_dim, critic_hidden_layers_dim, action_dim).to(device)
        self.q_opt = optim.RMSprop(self.q_net.parameters(), lr=critic_lr)
        self.q_cost_func = nn.MSELoss()
        
        self.critic = valueNet(state_dim, critic_hidden_layers_dim, action_dim).to(device)
        self.v_opt = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, continue_action_flag=A2C_kwargs.get('continue_action_flag', False)).to(device)
        self.a_opt = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
    
    def train(self):
        self.q_net.train()
        self.critic.train()
        self.actor.train()
    
    def eval(self):
        self.q_net.eval()
        self.critic.eval()
        self.actor.eval()
    
    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        ac_dist = self.actor.get_dist(state)
        a = ac_dist.sample()
        return a.detach().cpu().numpy()[0], ac_dist.log_prob(a).detach().cpu().numpy()

    def get_value(self, state, a=None):
        q_t = self.q_net(state)
        ac_dist = self.actor.get_dist(state)
        if a is None:
            return (q_t * ac_dist.probs).sum(1, keepdim=True)
        return ac_dist.probs, q_t, (q_t * ac_dist.probs).sum(1, keepdim=True), ac_dist.log_prob(a), ac_dist.entropy().mean()

    def acer_update(self, samples_buffer, wandb = None, update_lr=True):
        state, action, reward, next_state, done, sample_log_p = zip(*samples)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = np.stack(reward).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = np.stack(done).to(self.device)
        
        act_prob, q_t, v_t, log_prob, ac_entropy = self.get_value(state, action)
        rho = act_prob/(sample_log_p + 1e-10) 
        q_i = q_t.gather(1, action)
        rho_i = rho.gather(1, action)
        with torch.no_grad():
            next_v = get_value(next_state)
        q_retraces = self.compute_q_ret(reward, done, v_t, q_i, rho_i.clamp(max=self.clip), next_v)
        adv = q_retraces - v_t
        
        # value net loss 
        self.q_opt.zero_grad()
        loss = self.q_cost_func(q_retraces.float(), q_t.float())
        loss.backward()
        self.q_opt.step()
        
        # \partial log(\pi(a|s, \theta))}{\partial \theta} A
        a, log_prob = self.actor(state)
        actor_loss = -( (rho_i.clamp(max=self.clip) * adv).detach() * log_prob ).mean()
        
        # Bias correction for truncation
        adv_bc = (q_t - v_t)
        logf_bc = (act_prob + 1e-10).log()
        gain_bc = logf_bc * ((adv_bc * (1.0 - (self.clip / (rho + 1e-10))).clamp(min=0) * act_prob).detach())
        actor_loss = actor_loss - gain_bc.sum(-1).mean() - ac_entropy
        
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()

    def compute_q_ret(self, r, masks, values, q_i, rho_i, next_v):
        num_steps, num_processes = r.shape[:2]
        q_ret_list = r.new(num_steps + 1, num_processes, 1).zero_()
        q_ret_list[-1] = next_v
        for st in reversed(range(r.size(0))):
            # r_t + \gamma \hat{\rho}_{t+1}[Q^{ret}(x_{t+1}, a_{t+1}) - Q(x_{t+1}, a_{t+1})] + \gamma V(x_{t+1})
            q_ret_ = r[st] + self.gamma * q_ret_list[step + 1] * masks[step+1]
            q_ret_list[st] = q_ret_
            q_ret_ = rho_i[st] * (q_ret_list[st] - q_i[st]) + values[st]
        return q_retraces[:-1]

    def update(self, samples_buffer, wandb = None, update_lr=True):
        state, action, reward, next_state, done, actor_log_prob = zip(*samples_buffer)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = torch.LongTensor(np.stack(reward)).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = np.stack(done) # .to(self.device)
        R = self.trajectory_advantage(reward, state, done, action)
        d_set = memDataset(state, action, actor_log_prob, R)
        train_loader = DataLoader(
            d_set,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True
        )

        for mini_ep in range(self.k_epochs):
            for state_, action_, old_log_prob, R_ in train_loader:
                new_dist = self.actor.get_dist(state_)
                new_log_prob = new_dist.log_prob(action_)
                imp = torch.exp(new_log_prob - old_log_prob.detach())
                
                adv = R_ - self.critic(state_)
                actor_loss = -((imp * adv).detach() * new_log_prob).mean()

                self.v_opt.zero_grad()
                loss = 0.5 * (adv ** 2).mean()
                loss.backward()
                self.v_opt.step()

                self.a_opt.zero_grad()
                actor_loss.backward()
                self.a_opt.step()
                
    def trajectory_advantage(self, reward, state, done, actions):
        k_step = len(reward) - 1
        R = torch.zeros_like(reward)
        r = 0
        for t in reversed(range(k_step)):
            r = reward[t] + self.gamma * r * (1 - done[t])
            R[t] = r
        # adv = R - self.critic(state[0])
        return R 



class A2C:
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
        self.ent_coef = A2C_kwargs.get('ent_coef', 0.0)
        self.critic = valueNet(state_dim, critic_hidden_layers_dim, action_dim).to(device)
        self.v_opt = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.v_cost_func = nn.MSELoss()
        
        self.actor = policyNet(state_dim, actor_hidden_layers_dim, action_dim, continue_action_flag=A2C_kwargs.get('continue_action_flag', False)).to(device)
        self.a_opt = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
    
    def policy(self, state):
        state = torch.FloatTensor(np.array([state])).to(self.device)
        action_dist = self.actor.get_dist(state, self.action_bound)
        action = action_dist.sample()
        action = self._action_fix(action)
        return action.cpu().detach().numpy()

    def _one_deque_pp(self, samples):
        state, action, reward, next_state, done = zip(*samples)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.FloatTensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        if self.reward_func is not None:
            reward = self.reward_func(reward)

        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)

        old_v = self.critic(state)
        advantage = self.k_step_advantage(reward, state).to(self.device)
        action_dists = self.actor.get_dist(state, self.action_bound)
        log_probs = action_dists.log_prob(self._action_return(action))
        try:
            entropy_loss = action_dists.entropy().sum(1).mean()
        except Exception as e:
            # sperate action
            entropy_loss = action_dists.entropy().mean()
        if len(log_probs.shape) == 2:
            log_probs = log_probs.sum(dim=1)
        return state, action, log_probs, advantage, entropy_loss

    def update(self, samples_buffer, wandb = None, update_lr=True):
        state, action, log_probs, advantage, entropy_loss = self._one_deque_pp(samples_buffer)
        v_loss = (advantage ** 2).mean()
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        advantage = (advantage - advantage.mean()) / (torch.std(advantage) + 1e-8)
        actor_loss = -(advantage * log_probs).mean() - self.ent_coef * entropy_loss
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()

    def k_step_advantage(self, reward, state):
        k_step = len(reward) - 1
        r = 0
        for t in reversed(range(k_step)):
            r += self.gamma * reward[t] 
        
        r += r**k_step * self.critic(state[-1])
        
        return r - self.critic(state[0])

    def _action_return(self, act):
        if not self.continue_action_flag:
            return act.reshape(-1)
        if self.dist_type == 'beta' and self.continue_action_flag:
            # low ~ high -> 0-1 
            act_out = (act - self.action_low) / (self.action_high - self.action_low)
            return (act_out * 1 + 0).clip(1e-4, 9.999)
        return act

    def _action_fix(self, act):
        if not self.continue_action_flag:
            return act
        if self.dist_type == 'beta':
            # beta 0-1 -> low ~ high
            return act * (self.action_high - self.action_low) + self.action_low
        return act