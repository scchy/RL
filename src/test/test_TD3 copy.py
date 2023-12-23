
import os
from os.path import dirname
import sys
import gymnasium as gym
import torch

try:
    dir_ = dirname(dirname(__file__))
except Exception as e:
    dir_ = dirname(dirname('__file__'))

if len(dir_) == 0:
    dir_ = os.getcwd() + '/src'

print(dir_)
sys.path.append(dir_)
from RLAlgo.TD3 import TD3
from RLUtils import train_off_policy, play, Config, gym_env_desc
import numpy as np
from RLUtils.env_wrapper import FrameStack, CarV2SkipFrame, GrayScaleObservation, ResizeObservation


def reward_func(r, d):
    if r <= -100:
        r = -1
        d = True
    else:
        d = False
    return r, d


def CarRacing_TD3_test():
    """
    policyNet: 
    valueNet: 
    reference: https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html
    """
    env_name = 'CarRacing-v2'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(CarV2SkipFrame(env, skip=5)), 
            shape=84
        ), 
        num_stack=4
    )
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'TD3_CarRacing-v2_test2-2'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[256],
        critic_hidden_layers_dim=[256],
        # agent参数
        #    train_without_seed=True
        actor_lr=7.5e-5, # 1e-4,
        critic_lr=1.5e-3, #2.5e-3, # 3e-3,
        gamma=0.99,
        # 训练参数
        num_episode=50000,
        sample_size=128,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=1024*100,  
        off_minimal_size=512,
        max_episode_rewards=50000,
        max_episode_steps=1200, # 200
        # agent 其他参数
        TD3_kwargs={
            'CNN_env_flag': 1,
            'pic_shape': env.observation_space.shape,
            "env": env,
            'action_low': env.action_space.low,
            'action_high': env.action_space.high,
            # soft update parameters
            'tau': 0.05, 
            # trick2: Delayed Policy Update
            'delay_freq': 1,
            # trick3: Target Policy Smoothing
            'policy_noise': 0.2,
            'policy_noise_clip': 0.5,
            # exploration noise
            'expl_noise': 0.5,
            # 探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t
            'expl_noise_exp_reduce_factor': 1 - 1e-5
        }
    )
    agent = TD3(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        TD3_kwargs=cfg.TD3_kwargs,
        device=cfg.device
    )
    # 载入再学习
    agent.load_model(cfg.save_path)
    agent.eval()
    env = gym.make(env_name, render_mode='human') # 
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(CarV2SkipFrame(env, skip=5)), 
            shape=84
        ), 
        num_stack=4
    )
    play(env, agent, cfg, episode_count=2)


if __name__ == '__main__':
    # BipedalWalkerHardcore_TD3_test()
    # test_env()
    CarRacing_TD3_test()
    

