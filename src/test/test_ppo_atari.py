import os
from os.path import dirname
import sys
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ClipAction, NormalizeObservation, TransformObservation, NormalizeReward, TransformReward  
import torch
try:
    dir_ = dirname(dirname(__file__))
except Exception as e:
    dir_ = dirname(dirname('__file__'))

if len(dir_) == 0:
    dir_ = os.getcwd() + '/src'

print(dir_)
sys.path.append(dir_)
from RLAlgo.PPO2 import PPO2
from RLUtils import play, Config, gym_env_desc, ppo2_train
from RLUtils import make_atari_env


def DemonAttack_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/DemonAttack-v5' 
    env_name_str = env_name.replace('/', '-')
    num_envs = 10
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    envs = gym.vector.SyncVectorEnv(
        [make_atari_env(env_name) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-2'), 
        seed=202404,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[200],
        critic_hidden_layers_dim=[200],
        # agent参数
        actor_lr=1.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=1800, 
        off_buffer_size=500, # batch_size = off_buffer_size * num_env
        max_episode_steps=500,
        PPO_kwargs={
            'cnn_flag': True,
            'continue_action_flag': False,

            'lmbda': 0.95,
            'eps': 0.2,
            'k_epochs': 2,  # update_epochs
            'sgd_batch_size': 256, # 256, # 128,  # 1024, # 512, # 256,  
            'minibatch_size': 200, # 200, # 32,  # 900,  # 256, # 128,  
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.0,
            'max_grad_norm': 0.5, 
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False,
            'num_episode': 3000
        }
    )
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
    # {'P25': 0.054308490827679634, 'P50': 10.356741905212402, 'P75': 803.9899291992188, 'P95': 3986.836511230468, 'P99': 8209.22970703126}
    agent = PPO2(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=None, # lambda r: (r + 10.0)/10.0
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # {'P25': 0.07006523385643959, 'P50': 2.732730984687805, 'P75': 43.27479934692383, 
    # # 'P95': 330.70052642822253, 'P99': 460.6548526000974}
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_atari_env(env_name, clip_reward=False)() #, render_mode='human')()
    # env = make_atari_env(env_name, clip_reward=True)()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=3, play_without_seed=False, render=False, ppo_train=True)



def AirRaid_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/AirRaid-v5' 
    env_name_str = env_name.replace('/', '-')
    num_envs = 10
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    envs = gym.vector.SyncVectorEnv(
        [make_atari_env(env_name) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-2'), 
        seed=202404,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[200],
        critic_hidden_layers_dim=[200],
        # agent参数
        actor_lr=1.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=1800, 
        off_buffer_size=500, # batch_size = off_buffer_size * num_env
        max_episode_steps=500,
        PPO_kwargs={
            'cnn_flag': True,
            'continue_action_flag': False,

            'lmbda': 0.95,
            'eps': 0.2,
            'k_epochs': 2,  # update_epochs
            'sgd_batch_size': 256, # 256, # 128,  # 1024, # 512, # 256,  
            'minibatch_size': 200, # 200, # 32,  # 900,  # 256, # 128,  
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.0,
            'max_grad_norm': 0.5, 
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False,
            'num_episode': 3000
        }
    )
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
    # {'P25': 0.054308490827679634, 'P50': 10.356741905212402, 'P75': 803.9899291992188, 'P95': 3986.836511230468, 'P99': 8209.22970703126}
    agent = PPO2(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=None, # lambda r: (r + 10.0)/10.0
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # {'P25': 0.07006523385643959, 'P50': 2.732730984687805, 'P75': 43.27479934692383, 
    # # 'P95': 330.70052642822253, 'P99': 460.6548526000974}
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_atari_env(env_name, clip_reward=False, render_mode='human')()
    # env = make_atari_env(env_name, clip_reward=False)()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True, ppo_train=True)



def Alien_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/Alien-v5' 
    env_name_str = env_name.replace('/', '-')
    num_envs = 10
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    action_map = {
        0: 0, # NOOP
        1: 1, # FIRE
        2: 10, # UP
        3: 11, # RIGHT
        4: 12, # LEFT
        5: 13, # DOWN
        6: 14, # UPRIGHT
        7: 15, # UPLEFT
        8: 16, # DOWNRIGHT
        9: 17, # DOWNLEFT
        10: 10, # UPFIRE
        11: 11, # RIGHTFIRE
        12: 12, # LEFTFIRE
        13: 13, # DOWNFIRE
        14: 14, # UPRIGHTFIRE
        15: 15, # UPLEFTFIRE
        16: 16, # DOWNRIGHTFIRE
        17: 17 # DOWNLEFTFIRE
    }
    envs = gym.vector.SyncVectorEnv(
        [make_atari_env(env_name, clip_reward=True, action_map=action_map) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-1'), 
        seed=202404,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[240],
        critic_hidden_layers_dim=[240],
        # agent参数
        actor_lr=5.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=3600, 
        off_buffer_size=120, # batch_size = off_buffer_size * num_env
        max_episode_steps=120,
        PPO_kwargs={
            'cnn_flag': True,
            'continue_action_flag': False,

            'lmbda': 0.9,
            'eps': 0.25,
            'k_epochs': 2,  # update_epochs
            'sgd_batch_size': 128,
            'minibatch_size': 32, 
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.0,
            'max_grad_norm': 0.5, 
            'clip_vloss': True,
            'max_pooling': False,
            # 'min_adv_norm': True,

            'anneal_lr': False,
            'num_episode': 3000
        }
    )
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
    agent = PPO2(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_atari_env(env_name, clip_reward=True, action_map=action_map, render_mode='human')()
    # env = make_atari_env(env_name, clip_reward=False, action_map=action_map)()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True, ppo_train=True)


if __name__ == '__main__':
    # DemonAttack_v5_ppo2_test() # 2024-04-25
    # AirRaid_v5_ppo2_test()
    Alien_v5_ppo2_test()

