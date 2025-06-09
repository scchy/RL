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
from torch import multiprocessing as mp # cpu
try:
    dir_ = dirname(dirname(__file__))
except Exception as e:
    dir_ = dirname(dirname('__file__'))

if len(dir_) == 0:
    dir_ = os.getcwd() + '/src'

print(dir_)
sys.path.append(dir_)
from RLAlgo.A2C import SEA2CER 
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
    env = gym.wrappers.RecordEpisodeStatistics(env)
    action_contiguous_ = False # 是否将连续动作离散化
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'A2CER_A2C_{env_name_str}-1'),  
        # 网络参数 Atria-CNN + MLP
        hidden_layers_dim=[64, 128],
        actor_hidden_layers_dim=None, 
        critic_hidden_layers_dim=[32],
        # agent参数
        seed=202505,
        learning_rate=4.5e-4, 
        gamma=0.99,
        # 训练参数
        num_step_chunks=2600,
        step_chunk_size=20,
        off_buffer_size=5,
        max_episode_steps=300, 
        sample_size=64,
        T=1.05,
        A2C_kwargs={
            'k_epochs': 3, # 3, 
            'minibatch_size': 64, 
            'act_type': 'relu',
            'max_grad_norm': 0.5,
            'entroy_coef': None,
            'env_action_type': 'Cat',
            'normalize_adv': False
        }
    )
    agent = SEA2CER(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        A2C_kwargs=cfg.A2C_kwargs,
        device=cfg.device
    )
    # agent.train()
    # acer_train_off_policy(
    #     env, agent, cfg, 
    #     wandb_flag=False, 
    #     wandb_project_name=f"A2CER_A2C_{env_name_str}",
    #     train_without_seed=True, 
    #     test_ep_freq=3000, 
    #     test_episode_count=10
    # )
    agent.load_model(cfg.save_path)
    agent.eval()
    env = gym.make(env_name, render_mode='human')
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=False, ppo_train=True)


def A2CER_PL_test():
    # hogwild
    env_name = 'CartPole-v1' 
    env_name_str = env_name
 
    gym_env_desc(env_name)
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'A2CER_A2C_{env_name_str}-1'),  
        # 网络参数 Atria-CNN + MLP
        hidden_layers_dim=[128, 128],
        actor_hidden_layers_dim=None, 
        critic_hidden_layers_dim=[64],
        # agent参数
        seed=202505,
        learning_rate=4.5e-4, 
        gamma=0.99,
        # 训练参数
        num_step_chunks=220,
        step_chunk_size=24,
        off_buffer_size=3,
        max_episode_steps=300, 
        sample_size=64,
        T=1.05,
        A2C_kwargs={
            'k_epochs': 3, # 3, 
            'minibatch_size': 32, 
            'act_type': 'relu',
            'max_grad_norm': 0.5,
            'entroy_coef': None,
            'env_action_type': 'Cat',
            'normalize_adv': False
        }
    )
    cfg.device = 'cpu'
    cfg.num_processes = 10
    agent = SEA2CER(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        A2C_kwargs=cfg.A2C_kwargs,
        device=cfg.device
    )
    agent.train()
    mp.set_start_method('spawn', force=True)
    agent.actor_critic.share_memory()  
    processes = []
    for rank in range(cfg.num_processes):
        p = mp.Process(
            target=acer_train_off_policy, 
            kwargs=dict(
                env=env, agent=agent, cfg=cfg, 
                wandb_flag=False, 
                wandb_project_name=f"A2CER_A2C_{env_name_str}",
                train_without_seed=True, 
                test_ep_freq=3000, 
                test_episode_count=10,
                rank=rank
            )
        )
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    agent.load_model(cfg.save_path)
    agent.eval()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=False, ppo_train=True)



def A2CER_LunarLander_test():
    env_name = 'LunarLander-v2' 
    env_name_str = env_name
    gym_env_desc(env_name)
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    action_contiguous_ = False # 是否将连续动作离散化
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'A2CER_A2C_{env_name_str}-1'),  
        # 网络参数 Atria-CNN + MLP
        hidden_layers_dim=[128, 256],
        actor_hidden_layers_dim=[32], 
        critic_hidden_layers_dim=[64],
        # agent参数
        seed=202505,
        learning_rate=1.25e-3,
        gamma=0.99,
        # 训练参数
        num_step_chunks=6800,
        step_chunk_size=64,
        off_buffer_size=3,
        max_episode_steps=160, 
        sample_size=256,
        T=1.0,
        weight_sample=False,
        A2C_kwargs={
            'k_epochs': 3,  
            'minibatch_size': 64,
            'act_type': 'relu',
            'max_grad_norm': 0.5,
            'critic_coef': 1.02,
            'entroy_coef': 0.125,  # 135
            'env_action_type': 'Cat',
            'normalize_adv': True
        }
    )
    agent = SEA2CER(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        A2C_kwargs=cfg.A2C_kwargs,
        device=cfg.device
    )
    # agent.train()
    # acer_train_off_policy(
    #     env, agent, cfg, 
    #     wandb_flag=False, wandb_project_name=f"A2CER_A2C_{env_name_str}",
    #     train_without_seed=True, 
    #     test_ep_freq=3000, 
    #     test_episode_count=10,
    #     update_entroy_coef=False
    # )
    agent.load_model(cfg.save_path)
    agent.eval()
    cfg.max_episode_steps = 1620 
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=False, ppo_train=False)



if __name__ == '__main__':
    # A2CER_test() 
    A2CER_PL_test()
    # A2CER_LunarLander_test()

