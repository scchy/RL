# python3
# Author: Scc_hy
# Create Date: 2023-05-04
# Func: TD3
# ============================================================================
import typing as typ
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
from ._base_net import TD3ValueNet as valueNet
from ._base_net import TD3PolicyNet as policyNet
from ._base_net import TD3CNNPolicyNet as cnnPolicyNet
from ._base_net import TD3CNNValueNet  as cnnValueNet


def picTrans(pic_np):
    pic_trans = transforms.Compose([
            transforms.Grayscale(),
            # transforms.Lambda(lambda x:cv2.Canny(np.array(x), 170, 300)),
            # transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            # 转成tensor 作为模型输入
            transforms.ToTensor()
        ])
    if isinstance(pic_np, list) or isinstance(pic_np, tuple) :
        out_p = [pic_trans(Image.fromarray(i)) for i in pic_np]
        out = torch.stack(out_p, dim=0)
        return out
    return pic_trans(Image.fromarray(pic_np))


class TD3:
    def __init__(
        self,
        state_dim: int, 
        actor_hidden_layers_dim: typ.List, 
        critic_hidden_layers_dim: typ.List, 
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        TD3_kwargs: typ.Dict,
        device: torch.device=None
    ):
        """
        state_dim (int): 环境的sate维度  
        actor_hidden_layers_dim (typ.List): actor hidden layer 维度  
        critic_hidden_layers_dim (typ.List): critic hidden layer 维度  
        action_dim (int): action的维度  
        actor_lr (float): actor学习率  
        critic_lr (float): critic学习率  
        gamma (float): 折扣率  
        TD3_kwargs (typ.Dict): TD3算法的三个trick的输入  
            example:  
                TD3_kwargs={  
                    'action_low': env.action_space.low[0],  
                    'action_high': env.action_space.high[0],  
                - soft update parameters  
                    'tau': 0.005,   
                - trick2: Target Policy Smoothing  
                    'delay_freq': 1,  
                - trick3: Target Policy Smoothing  
                    'policy_noise': 0.2,  
                    'policy_noise_clip': 0.5,  
                - exploration noise  
                    'expl_noise': 0.25,  
                    -  探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t  
                    'expl_noise_exp_reduce_factor': 0.999  
                }  
        device (torch.device): 运行的device  
        """
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.action_low = TD3_kwargs.get('action_low', -1.0)
        self.action_high = TD3_kwargs.get('action_high', 1.0)
        print(self.action_low, self.action_high)
        self.CNN_env_flag = TD3_kwargs['CNN_env_flag']
        self.max_action = max(
            max(abs(self.action_low)) if isinstance(self.action_low, np.ndarray) else self.action_low, 
            max(abs(self.action_high)) if isinstance(self.action_high, np.ndarray) else self.action_high
        )
        policy = cnnPolicyNet if self.CNN_env_flag else policyNet 
        value_net = cnnValueNet if self.CNN_env_flag else valueNet 
        # if self.CNN_env_flag:
        #     self.feat_extractor = stateFeatureExtractor()
        #     self.feat_extractor.to(device)
        #     self.target_feat_extractor = copy.deepcopy(self.feat_extractor)
        #     self.target_feat_extractor.to(device)

        self.actor = policy(
            state_dim, 
            actor_hidden_layers_dim, 
            action_dim, 
            action_bound = TD3_kwargs['env'] if  "env" in TD3_kwargs else self.max_action,
            state_feature_share=self.CNN_env_flag
        )
        self.critic = value_net(
            state_dim,
            action_dim, 
            critic_hidden_layers_dim,
            state_feature_share=self.CNN_env_flag
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.tau = TD3_kwargs.get('tau', 0.01)
        self.policy_noise = TD3_kwargs.get('policy_noise', 0.2) 
        self.policy_noise_clip_low = -TD3_kwargs.get('policy_noise_clip', 0.5)
        self.policy_noise_clip_high = TD3_kwargs.get('policy_noise_clip', 0.5)
        if isinstance(self.action_low, np.ndarray):
            self.policy_noise_clip_low = TD3_kwargs.get('policy_noise_clip', 0.5) * self.action_low
            self.policy_noise_clip_high = TD3_kwargs.get('policy_noise_clip', 0.5) * self.action_high
            
        self.expl_noise = TD3_kwargs.get('expl_noise', 0.25)
        self.expl_noise_exp_reduce_factor = TD3_kwargs.get('expl_noise_exp_reduce_factor', 1)
        self.delay_counter = -1
        # actor延迟更新的频率: 论文建议critic更新2次， actor更新1次， 即延迟1次
        self.delay_freq = TD3_kwargs.get('delay_freq', 1)
        
        # Normal sigma
        self.train_noise = self.expl_noise
        self.update_count = 0
        self.training = False

    @torch.no_grad()
    def smooth_action(self, state):
        """
        trick3: Target Policy Smoothing
            在target-actor输出的action中增加noise
        """
        pt_action_low = torch.FloatTensor(self.action_low)
        pt_action_high = torch.FloatTensor(self.action_high)
        act_target = self.target_actor(state).cpu()
        try:
            # self.action_low + self.action_high)/2
            noise = torch.stack([torch.normal(np.zero_like(self.action_low), (self.action_high-self.action_low)) for _ in range(act_target.shape[0])])
            noise = (noise * self.policy_noise).clip(
                torch.FloatTensor(self.policy_noise_clip_low), 
                torch.FloatTensor(self.policy_noise_clip_high)
            )
        except Exception as e:
            noise = (torch.randn(act_target.shape).float() * self.policy_noise).clip(
                torch.FloatTensor(self.policy_noise_clip_low), 
                torch.FloatTensor(self.policy_noise_clip_high))
        smoothed_target_a = (act_target + noise).clip(pt_action_low, pt_action_high)
        return smoothed_target_a

    def random_action(self):
        return np.random.uniform(self.action_low, self.action_high)
    
    def train_action(self, act):
        try:
            # self.action_low + self.action_high)/2
            action_noise = torch.normal(np.zero_like(self.action_low), (self.action_high-self.action_low)).numpy()  * self.train_noise
        except Exception as e:
            action_noise = np.random.randn(self.action_dim) * self.train_noise

        self.train_noise *= self.expl_noise_exp_reduce_factor
        self.train_noise = np.max([0.01, self.train_noise])
        # print("action_noise=", np.round(action_noise, 3), "act=", np.round(act, 3))
        return (act.detach().cpu().numpy()[0] + action_noise).clip(self.action_low, self.action_high)

    @torch.no_grad()
    def policy(self, state):
        try:
            state = torch.FloatTensor(state[np.newaxis, ...]).to(self.device)
        except Exception as e:
            state = torch.stack(state._frames).float().to(self.device)

        # state = picTrans(state).to(self.device)
        # if self.CNN_env_flag:
        #     # state /= 255.0
        #     state = self.feat_extractor(state)
        act = self.actor(state)
        if self.training:
            return self.train_action(act)
        return act.detach().cpu().numpy()[0].clip(self.action_low, self.action_high)

    def update(self, samples):
        self.delay_counter += 1
        state, action, reward, next_state, done = zip(*samples)
        # state = torch.FloatTensor(picTrans(state)).to(self.device)
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.tensor(np.stack(action)).to(self.device)
        reward = torch.tensor(np.stack(reward)).view(-1, 1).to(self.device)
        # next_state = torch.FloatTensor(picTrans(next_state)).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(np.stack(done)).view(-1, 1).to(self.device)
        # if self.CNN_env_flag:
        #     # state /= 255.0
        #     state = self.feat_extractor(state)
        #     # next_state /= 255.0
        #     next_state = self.target_feat_extractor(next_state)

        # 计算目标Q
        smooth_act = self.smooth_action(next_state).to(self.device)
        # trick1: **Clipped Double Q-learning**: critic中有两个`Q-net`, 每次产出2个Q值，使用其中小的
        target_Q1, target_Q2 = self.target_critic(next_state, smooth_act)
        target_Q = torch.minimum(target_Q1, target_Q2)
        target_Q = reward + (1.0 - done) * self.gamma * target_Q
        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(state, action)
        q_loss = F.mse_loss(current_Q1.float(), target_Q.float().detach()) + F.mse_loss(current_Q2.float(), target_Q.float().detach())
        self.critic_opt.zero_grad()
        q_loss.backward()
        for n, p in self.critic.q1_cnn_feature[0].named_parameters():
            g_sum = p.grad.sum()
            if g_sum > -0.05 and g_sum < 0.05:
                print(f"\nq_loss={q_loss}  self.critic.q1_cnn_feature[0] grad.sum={g_sum}")
            break
        self.critic_opt.step()
        
        # trick2: **Delayed Policy Update**: actor的更新频率要小于critic(当前的actor参数可以产出更多样本)。
        if self.delay_counter == self.delay_freq:
            # actor 延迟update
            ac_action = self.actor(state.detach())
            actor_loss = -torch.mean(self.critic.Q1(state.detach(), ac_action))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)
            # self.soft_update(self.feat_extractor, self.target_feat_extractor)
            self.delay_counter = -1

    def save_model(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        act_f = os.path.join(file_path, 'TD3_actor.ckpt')
        critic_f = os.path.join(file_path, 'TD3_critic.ckpt')
        torch.save(self.actor.state_dict(), act_f)
        torch.save(self.critic.state_dict(), critic_f)
        # if self.CNN_env_flag:
        #     feat_f = os.path.join(file_path, 'TD3_feat.ckpt')
        #     torch.save(self.feat_extractor.state_dict(), feat_f)

    def load_model(self, file_path):
        act_f = os.path.join(file_path, 'TD3_actor.ckpt')
        critic_f = os.path.join(file_path, 'TD3_critic.ckpt')
        self.actor.load_state_dict(torch.load(act_f, map_location='cpu'))
        self.critic.load_state_dict(torch.load(critic_f, map_location='cpu'))
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # if self.CNN_env_flag:
        #     feat_f = os.path.join(file_path, 'TD3_feat.ckpt')
        #     self.feat_extractor.load_state_dict(torch.load(feat_f))

    def train(self):
        self.training = True
        # if self.CNN_env_flag:
        #     self.feat_extractor.train()
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

    def eval(self):
        self.training = False
        # if self.CNN_env_flag:
        #     self.feat_extractor.eval()
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )