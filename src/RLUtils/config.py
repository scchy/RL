
import torch
import typing as typ
import numpy as np
import random
import os


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


class Config:  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(
        self, 
        env,  
        save_path: typ.AnyStr = r'D:\TMP\new_model.ckpt', 
        split_action_flag: bool=False,
        action_dim: int=20,
        hidden_layers_dim: typ.List[int] = [10, 10],
        actor_hidden_layers_dim: typ.List[int] = [10, 10],
        critic_hidden_layers_dim: typ.List[int] = [10, 10],
        learning_rate: float = 2e-3,
        actor_lr: float = 3e-5,
        critic_lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon: float = 0.01,
        target_update_freq: int = 3,
        num_episode: int = 100,
        max_episode_rewards: float = 260*100000,
        max_episode_steps:int = 260,
        render: bool = False,
        off_buffer_size: int = 2048,
        off_minimal_size: int = 1024,
        sample_size: int = 128,
        seed: int = 42,
        **kwargs
    ):
        """
        env (_type_): 智能体环境
        save_path (_type_, optional): actor或者target_q的模型存储地址. 默认:  r'D:\TMP\new_model.ckpt'.  
        split_action_flag (bool, optional): 是否将连续动作离散化. 默认:  False.  
        action_dim (int, optional): action的空间. 默认:  20.  
        hidden_layers_dim (typ.List[int], optional): 网络的每层大小. 默认:  [10, 10].  
        actor_hidden_layers_dim (typ.List[int], optional): actor网络的每层大小. 默认:  [10, 10].  
        critic_hidden_layers_dim (typ.List[int], optional): critic网络的每层大小. 默认:  [10, 10].  
        learning_rate (float, optional): 网络的学习率. 默认:  2e-3.  
        actor_lr (float, optional): actor网络的学习率. 默认:  3e-5.  
        critic_lr (float, optional): critic网络的学习率. 默认:  5e-4.  
        gamma (float, optional): 折扣率. 默认:  0.95.  
        epsilon (float, optional): epsilon-greedy 的探索系数. 默认:  0.01.  
        target_update_freq (int, optional): 目标网络的更新频率(DQN) 默认:  3.  
        num_episode (int, optional): _description_. 默认:  100.  
        max_episode_rewards (float, optional): 回合停止设置: 一个回合最大奖励数. 默认:  260.  
        max_episode_steps (int, optional): 回合停止设置: 一个回合最大步数. 默认:  260.  
        render (bool, optional): 是否显示. 默认:  False.  
        off_buffer_size (int, optional): off policy训练的时候样本池的大小. 默认:  2048.  
        off_minimal_size (int, optional): off policy训练的时候样本池最小多少时进行网络更新. 默认:  1024.  
        sample_size (int, optional): 样本抽样大小. 默认:  128.  
        kwargs 其他参数  
            如 
            dqn_type = 'duelingDQN'
            DDPG_kwargs = {
            }
        """
        # 智能体相关网络参数
        self.hidden_layers_dim = hidden_layers_dim
        self.critic_hidden_layers_dim = critic_hidden_layers_dim
        self.actor_hidden_layers_dim = actor_hidden_layers_dim
        self.learning_rate = learning_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # 智能体训练相关参数
        self.epsilon = epsilon # epsilon-greedy 探索策略参数
        self.gamma = gamma # 折扣系数
        self.target_update_freq = target_update_freq
        self.num_episode = num_episode
        self.render = render
        # 回合停止控制
        self.max_episode_rewards = max_episode_rewards
        self.max_episode_steps = max_episode_steps
        # off policy 训练 智能体更新buffer设置
        self.off_buffer_size = off_buffer_size
        self.off_minimal_size = off_minimal_size
        self.sample_size = sample_size
        
        # actor 或者 target_q的存储位置
        self.save_path = save_path
        
        # 环境相关变量
        if hasattr(env, 'num_envs'):
            env = env.env_fns[0]()
        self.action_dim = action_dim
        self.state_dim = env.observation_space.shape[0]
        if len(env.observation_space.shape) == 3:
            self.state_dim = env.observation_space.shape[-1]
        all_seed(seed)
        self.seed = seed
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            if not split_action_flag:
                self.action_dim = env.action_space.shape[0]
                print()
            pass
        print(f'device={self.device} | env={str(env)}(state_dim={self.state_dim}, action_dim={self.action_dim})')
        # 其他参数设置
        for k, v in kwargs.items():
            print(k, '=>', v)
            self.__setattr__(k, v)

        if self.off_buffer_size < self.off_minimal_size:
            self.off_minimal_size = self.off_buffer_size - 1

    # def __iter__(self):
    
    # def dict2config(self, dict):
        