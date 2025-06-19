# python3
# Author: Scc_hy
# Create Date: 2025-06-18
# Func: Soft Q-Learning
#       its structure resembles an actor-critic algorithm.
# reference: https://github.com/haarnoja/softqlearning/blob/master/softqlearning/algorithms/sql.py
# ============================================================================

from torch import Tensor
import torch 
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict
from copy import deepcopy

EPS = 1e-6


class NNQFunction(nn.Module):
    """ 
    reference: 
        https://github.com/haarnoja/softqlearning/blob/master/softqlearning/misc/nn.py
        MLPFunction & feedforward_net

    适用MuJoCo环境
    """ 
    def __init__(self, state_dim, action_dim, hidden_layer_sizes):
        super(NNQFunction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        common_dim = hidden_layer_sizes[0]
        self.state_linear = nn.Linear(state_dim, common_dim)
        self.action_linear = nn.Linear(action_dim, common_dim)
        self.common_relu = nn.ReLU(inplace=True)
        
        self.feature_nn = nn.ModuleList()
        for idx in range(1, len(hidden_layer_sizes)):
            self.feature_nn.append(
                nn.ModuleDict({
                    'linear': nn.Linear(hidden_layer_sizes[idx-1], hidden_layer_sizes[idx]),
                    'linear_active': nn.ReLU(inplace=True)
                })
            )
        
        self.head = nn.Linear(hidden_layer_sizes[-1], 1)
    
    def forward(self, state, action):
        x = self.common_relu(self.state_linear(state) + self.action_linear(action))
        for layer in self.feature_nn:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)



class StochasticNNPolicy(nn.Module):
    """ 
    Reference:
    ---------
        https://github.com/haarnoja/softqlearning/blob/master/softqlearning/policies/stochastic_policy.py
        StochasticNNPolicy & NNPolicy & feedforward_net
    """ 
    def __init__(self, state_dim, action_dim, hidden_layer_sizes, squash=True):
        super(StochasticNNPolicy, self).__init__()
        self.squash = squash
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.normal_dist = Normal(torch.zeros(action_dim), torch.ones(action_dim))
        common_dim = hidden_layer_sizes[0]
        self.state_linear = nn.Linear(state_dim, common_dim)
        self.action_linear = nn.Linear(action_dim, common_dim)
        self.common_relu = nn.ReLU(inplace=True)
        
        self.feature_nn = nn.ModuleList()
        for idx in range(1, len(hidden_layer_sizes)):
            self.feature_nn.append(
                nn.ModuleDict({
                    'linear': nn.Linear(hidden_layer_sizes[idx-1], hidden_layer_sizes[idx]),
                    'linear_active': nn.ReLU(inplace=True)
                })
            )
        
        self.head = nn.Linear(hidden_layer_sizes[-1], action_dim)

    def forward(self, state, n_action_samples=1):
        # \sim N(0, 1)
        n_samples = state.shape[0]
        
        if n_action_samples > 1:
            state = state[:, None, :]
            sample_size = (n_samples, n_action_samples)
        else:
            sample_size = (n_samples, )
        
        normal_action = self.normal_dist.sample(sample_size).to(state.device)
        x = self.common_relu(self.state_linear(state) + self.action_linear(normal_action))
        for layer in self.feature_nn:
            x = layer['linear_active'](layer['linear'](x))
        x = self.head(x)
        if self.squash:
            return torch.tanh(x)
        return x


@torch.no_grad()
# SVGD 
def adaptive_isotropic_gaussian_kernel(xs: Tensor, ys: Tensor, h_min: float=1e-3):
    """Gaussian kernel with dynamic bandwidth.
    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.
    
    Parameters
    ----------
    :param Tensor xs: (N, Kx, D) containing N sets of Kx particles of dimension D. This is the first kernel argument.
    :param Tensor ys: (N, Ky, D) containing N sets of Kx particles of dimension D. This is the second kernel argument.

    
    Returns
    ----------
    dict: { 'output': (N, Kx, Ky), 'gradient': (N, Kx, Ky, D) }
    - output: representing the kernel matrix for inputs `xs` and `ys`.
    - gradient: representing the gradient of the kernel with respect to `xs`
    
    Reference:
    ---------
        https://github.com/haarnoja/softqlearning/blob/master/softqlearning/misc/kernel.py

        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """ 
    Kx, D = xs.shape[-2:]
    Ky, D = ys.shape[-2:]
    leading_shape = torch.tensor(xs.shape[:-2])
    # Compute the pairwise distances of left and right particles.
    diff = xs.unsqueeze(-2) - ys.unsqueeze(-3)
    # ... x Kx x Ky x D
    dist_sq = torch.sum(diff**2, dim=-1, keepdim=False)
    
    # Get median.
    input_shape = torch.cat((leading_shape, torch.tensor([Kx * Ky])), dim=0) 
    values, _ = torch.topk(
        input=torch.reshape(dist_sq, input_shape.numpy().tolist()),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        sorted=True)
    medians_sq = values[..., -1]
    h = medians_sq / torch.tensor(Kx).log()
    h = torch.maximum(h, torch.tensor(h_min, dtype=torch.float32))
    
    h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)
    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky
    # Construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    kappa_expanded = kappa.unsqueeze(-1)
    
    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    return {"output": kappa, "gradient": kappa_grad}



# SQL 
class SoftQ:
    """Soft Q-learning (SQL).

    Reference:
    ------------
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """
    def __init__(
            self,
            state_dim, 
            action_dim,
            gamma=0.99,
            actor_hidden_layers_dim=[128, 128],
            critic_hidden_layers_dim=[128, 128],
            actor_lr=1E-3,
            critic_lr=1E-3,
            device='cuda',
            SQL_kwargs: Dict=dict(
                value_n_particles=16,
                kernel_n_particles=16,
                kernel_update_ratio=0.5,
                critcic_traget_update_freq=1000
            )
    ):
        self.kernel_fn = adaptive_isotropic_gaussian_kernel
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.value_n_particles = SQL_kwargs.get('value_n_particles', 16)
        self.kernel_n_particles = SQL_kwargs.get('kernel_n_particles', 16)
        self.kernel_update_ratio = SQL_kwargs.get('kernel_update_ratio', 0.5)
        self.critcic_traget_update_freq = SQL_kwargs.get('critcic_traget_update_freq', 1000)
        self.reward_scale = SQL_kwargs.get('reward_scale', 1)

        self.actor = StochasticNNPolicy(state_dim, action_dim, actor_hidden_layers_dim).to(device)
        self.critic = NNQFunction(state_dim, action_dim, critic_hidden_layers_dim).to(device)
        self.tar_critic = deepcopy(self.critic)
        
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)
        self.count = 0
        
    @torch.no_grad()
    def policy(self, state, n_action_samples=1):
        state = torch.FloatTensor([state]).to(self.device)
        act = self.actor(state) 
        # print(f'{state.shape=} {act.shape=}')
        return act.cpu().detach().numpy()[0]
    
    def update(self, samples):
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)

        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.tensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)
        
        self._create_td_update(state, action, reward, next_state, done)
        self._svgd_update(state)
        if self.count % self.critcic_traget_update_freq == 0:
            # print(f"critcic_traget_update {self.count} // {self.critcic_traget_update_freq}")
            self.tar_critic.load_state_dict(
                self.critic.state_dict()
            )

    def _svgd_update(self, state):
        act = self.actor(state, n_action_samples=self.kernel_n_particles)
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(self.kernel_n_particles * self.kernel_update_ratio)
        n_fixed_actions = self.kernel_n_particles - n_updated_actions
        fixed_actions, updated_actions = torch.split(
            act, 
            [n_fixed_actions, n_updated_actions], 
            dim=1
        )
        # eqution13: part_1-2 \nabla _{a^\prime} Q^\theta_{soft}(s_t, a^\prime)|_{a\prime=a_t}
        svgd_tar_values = self.critic(state[:, None, :], fixed_actions)
        # Target log-density. Q_soft in Equation 13:
        squash_corr = torch.sum(torch.log(1 - fixed_actions**2 + EPS), dim=-1)
        log_p = svgd_tar_values.squeeze(2) + squash_corr
        # 计算 log_p 对 fixed_actions 的梯度
        grad_log_p = torch.autograd.grad(
            log_p, fixed_actions, 
            grad_outputs=torch.ones_like(log_p), 
            create_graph=True
        )[0]
        grad_log_p = grad_log_p.unsqueeze(2).detach()
        # print(f'{grad_log_p.shape=} {grad_log_p.mean()=} {grad_log_p.min()=}')
        
        kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions)
        # Kernel function in Equation 13:
        kappa = torch.unsqueeze(kernel_dict["output"], dim=3)
        # eqution13: part_1 kappa * grad_log_p part_2 k_gradient
        action_gradients = torch.mean(kappa * grad_log_p + kernel_dict["gradient"], dim=1)
        # Propagate the gradient through the policy network (Equation 14).
        # 计算 gradients
        gradients = torch.autograd.grad(
            outputs=updated_actions,
            inputs=self.actor.parameters(),
            grad_outputs=action_gradients,
            create_graph=True
        )
        # 计算 surrogate_loss
        actor_loss = -sum(
            torch.sum(w * g.detach()) for w, g in zip(self.actor.parameters(), gradients)
        )
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

    def _create_td_update(self, state, action, reward, next_state, done):
        # 生成 [0, 1) 范围内的随机数
        random_actions = torch.rand(1, self.value_n_particles, self.action_dim).to(self.device)
        # 将范围调整到 [-1, 1)
        tar_action = 2 * random_actions - 1
        q_next = self.tar_critic(next_state[:, None, :], tar_action)
        # Equation 10:
        q_next = torch.logsumexp(q_next, dim=1) - torch.log(torch.tensor(self.value_n_particles).to(self.device))
        q_next += self.action_dim * np.log(2)
        
        q_tar = self.reward_scale * reward + self.gamma * q_next * (1 - done)
        q = self.critic(state, action)

        # # Equation 11:
        critic_loss = torch.mean(0.5 * (q_tar.detach() - q) ** 2)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

    def train(self):
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        self.actor.eval()
        self.critic.eval()




if __name__ == '__main__':
    qf = NNQFunction(18, 4, hidden_layer_sizes=[128, 128])
    actor = StochasticNNPolicy(18, 4, hidden_layer_sizes=[128, 128])
    state = torch.randn(10, 18)
    a = torch.randn(10, 4)
    
    tar_action = torch.rand(10, 6, 4)
    
    xs = torch.randn(20, 16, 4)
    ys = torch.randn(20, 16, 4)
    res_ = adaptive_isotropic_gaussian_kernel(xs, ys)
    
    print("qf(state, a).shape=", qf(state, a).shape)
    print("qf(state, tar_action).shape=", qf(state[:, None, :], tar_action).shape)
    print("actor(state, 10).shape=", actor(state, 10).shape)
    print("res_['output'].shape=, res_['gradient'].shape=",
          res_['output'].shape, res_['gradient'].shape)




