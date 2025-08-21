# python3
# Author: Scc_hy
# Create Date: 2025-08-20
# Func: decision transformer
# reference: https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/training/trainer.py
# ============================================================================

import typing as typ
import numpy as np
import os
import math
from gymnasium.wrappers.normalize import RunningMeanStd
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from .DTModules.DT_model import DecisionTransformer



class DTAgent:
    """
    DecisionTransformer
    Reference
    ----------
    - Github: https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
    - paper: [DT: Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345)
    """
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        K: int,
        max_ep_len: int,
        DT_kwargs: typ.Dict=dict(
            embed_dim=128,
            n_layer=3,
            n_head=1,
            activation_function='relu',
            dropout=0.1
        ),
        device: torch.device='cpu',
        reward_func: typ.Callable=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.max_grad_norm = DT_kwargs.get('max_grad_norm', 0.25)
        self.rtg_scale = DT_kwargs.get('rtg_scale', 1.0)
        self.device = device
        self.max_length = K
        self.model = DecisionTransformer(
            state_dim, 
            action_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=DT_kwargs['embed_dim'],
            n_layer=DT_kwargs['n_layer'],
            n_head=DT_kwargs['n_head'],
            n_inner=4 * DT_kwargs['embed_dim'],
            activation_function=DT_kwargs['activation_function'],
            n_positions=1024,
            resid_pdrop=DT_kwargs['dropout'],
            attn_pdrop=DT_kwargs['dropout'],
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.norm_obs = DT_kwargs.get('norm_obs', True)
        # NormalizeObservation
        self.obs_rms = RunningMeanStd(shape=(state_dim,))

    def normalize(self, obs, update_flag=True):
        if not self.norm_obs:
            return obs
        if update_flag:
            up_obs = obs.reshape(-1, self.state_dim)
            self.obs_rms.update(up_obs if isinstance(obs, np.ndarray) else up_obs.detach().cpu().numpy())
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-6)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def padding_batch(self, t, pad_num=1.0, final_dim=None):
        pad_nt = self.max_length - t.shape[1]
        if pad_nt == 0:
            return t 
        pad_tensor = torch.ones(1, pad_nt, dtypes=t.dtype) * pad_num
        if final_dim is not None:
            pad_tensor = torch.ones(1, pad_nt, final_dim, dtypes=t.dtype) * pad_num
        return torch.concat([pad_tensor.to(self.device), t])
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = torch.FloatTensor(self.normalize(states.detach().cpu().numpy(), False)).to(self.device)
        return self.model.get_action(states, actions, rewards, returns_to_go, timesteps, **kwargs)

    def update(self, batch_tensor, wandb_w=None):
        states, actions, reward, next_state, done, rtg, timesteps, attention_mask = batch_tensor
        states = self.normalize(states, True)
        # B, d -> B//K, K, d
        states = states.float().to(self.device)
        actions = actions.float().to(self.device)
        reward = reward.float().to(self.device)
        done = done.to(self.device)
        rtg = rtg.float().to(self.device)
        timesteps = timesteps.long().to(self.device)
        attention_mask = attention_mask.to(self.device)

        rtg /= self.rtg_scale
        action_target = torch.clone(actions)
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, reward, rtg, timesteps, attention_mask=attention_mask,
        )
        # note: currently indexing & masking is not fully correct
        # print(f"{state_preds.shape=}, {action_preds.shape=}, {reward_preds.shape=}")
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = 0.5 * (action_preds - action_target).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if wandb_w is not None:
            with torch.no_grad():
                log_dict = {
                    "action_error": torch.mean((action_preds-action_target)**2).detach().cpu().item(),
                }
            wandb_w.log(log_dict)
        return reward_preds.detach().cpu().numpy().mean() * self.rtg_scale 


    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        model_f = os.path.join(file_path, 'DT_model.ckpt')
        if self.norm_obs:
            self.model.obs_mean.data = torch.tensor(self.obs_rms.mean).to(self.device)
            self.model.obs_var.data = torch.tensor(self.obs_rms.var).to(self.device)
        
        torch.save(self.model.state_dict(), model_f)

    def load_model(self, file_path):
        model_f = os.path.join(file_path, 'DT_model.ckpt')
        actor_d = torch.load(model_f, map_location='cpu')
        self.model.load_state_dict(actor_d)

        self.obs_rms.mean = actor_d['obs_mean'].detach().cpu().numpy()
        self.obs_rms.var = actor_d['obs_var'].detach().cpu().numpy()
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)


