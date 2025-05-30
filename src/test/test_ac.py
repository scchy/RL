# python3
# Create Date: 2025-05-30
# Author: Scc_hy
# Func: AC A2C A3C Test
# ===============================================================================

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
from RLAlgo.A2C import A2CER 
from RLUtils.trainer import acer_train_off_policy
from RLUtils import play, Config, gym_env_desc, ppo2_train, ppo2_play
from RLUtils import make_atari_env, make_envpool_atria, spSyncVectorEnv


def A2CER_test():
    # num_episode=500 action_contiguous_ = False epsilon=0.01
    env_name = 'CartPole-v1' 
    env_name_str = env_name
    # 要需要提升探索率，steps需要足够大
    # num_episode=200 action_contiguous_ = False epsilon=0.05 max_episode_steps=500
    # env_name = 'MountainCar-v0' 
    gym_env_desc(env_name)
    env = gym.make(env_name)
    action_contiguous_ = False # 是否将连续动作离散化
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'A2CER_A2C_{env_name_str}-1'),  
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[200, 200], 
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        seed=202505,
        actor_lr=4.5e-3, 
        critic_lr=5.5e-3, 
        gamma=0.99,
        # 训练参数
        num_episode=100,
        off_minimal_size=3*50,
        off_buffer_size=3*50,
        max_episode_steps=300, 
        A2C_kwargs={
            'k_epochs': 3, # 3, 
            'minibatch_size': 20, 
            'act_type': 'relu',
            'dist_type': 'norm',
            'max_grad_norm': 1.5
        }
    )
    agent = A2CER(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        A2C_kwargs=cfg.A2C_kwargs,
        device=cfg.device
    )
    agent.train()
    acer_train_off_policy(
        env, agent, cfg, 
        wandb_flag=False, wandb_project_name=f"A2CER_A2C_{env_name_str}",
        train_without_seed=True, 
        test_ep_freq=cfg.off_buffer_size * 10, 
        test_episode_count=10
    )
    # agent.load_model(cfg.save_path)
    agent.eval()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=False, ppo_train=True)



if __name__ == '__main__':
    A2CER_test() 

