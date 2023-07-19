
import os
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.DDPG import DDPG
from RLUtils import train_off_policy, play, Config, gym_env_desc




def ddpg_test():
    """
    policyNet: 
    valueNet: 
    """
    # env_name = 'Pendulum-v1'
    # env = gym.make(env_name)
    # Pendulum_cfg = Config(
    #     env, 
    #     num_episode = 180,
    #     save_path=r'D:\TMP\ddpg_test_actor.ckpt', 
    #     actor_hidden_layers_dim=[128, 64],
    #     critic_hidden_layers_dim=[64, 32],
    #     actor_lr=3e-5,
    #     critic_lr=5e-4,
    #     sample_size=256,
    #     off_buffer_size=2048,
    #     off_minimal_size=1024,
    #     max_episode_rewards=2048,
    #     max_episode_steps=240,
    #     gamma=0.9,
    #     DDPG_kwargs={
    #         'tau': 0.05, # soft update parameters
    #         'sigma': 0.005, # noise
    #         'action_bound': 2.0
    #     }
    # )
    env_name = 'MountainCarContinuous-v0'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    cfg = Config(
        env, 
        # 环境参数
        save_path=r'D:\TMP\ddpg_MountainCarContinuous_test_actor.ckpt', 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[512, 256],
        critic_hidden_layers_dim=[512, 256],
        # agent参数
        actor_lr=3e-4,
        critic_lr=5e-3,
        gamma=0.99,
        # 训练参数
        num_episode=100,
        sample_size=512,
        off_buffer_size=2048000,
        off_minimal_size=2048*5,
        max_episode_rewards=90,
        max_episode_steps=1500,
        # agent 其他参数
        DDPG_kwargs={
            'tau': 0.005, # soft update parameters
            'sigma': 0.8, # noise
            'action_bound': 1.1,
            'action_low': env.action_space.low[0],
            'action_high': env.action_space.high[0],
        }
    )
    agent = DDPG(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        DDPG_kwargs=cfg.DDPG_kwargs,
        device=cfg.device
    )
    # agent.train = True
    # train_off_policy(env, agent, cfg, done_add=True)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    agent.train = False
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=1)



def LunarLanderContinuous_ddpg_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'LunarLanderContinuous-v2'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'ddpg_LunarLanderContinuous-v2_test_actor-3.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 128],
        critic_hidden_layers_dim=[128, 128],
        # agent参数
        # actor_lr=5e-4, # 128, 64
        # critic_lr=1e-3, # 128, 64
        actor_lr=1e-4,  
        critic_lr=5e-4,  
        gamma=0.99,
        # 训练参数
        num_episode=5000,
        sample_size=256,
        off_buffer_size=165*100, # 环境不是非常复杂
        off_minimal_size=1024,
        max_episode_rewards=500,
        max_episode_steps=180,  # 215 step get-299 rewards # 165
        # agent 其他参数
        DDPG_kwargs={
            'tau': 0.001, # soft update parameters
            'sigma': 0.15, # noise
            'sigma_exp_reduce_factor': 1,
            'action_low': env.action_space.low,
            'action_high': env.action_space.high,
            'off_minimal_size': 1024
        }
    )
    agent = DDPG(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        DDPG_kwargs=cfg.DDPG_kwargs,
        device=cfg.device
    )
    # agent.train = True
    # train_off_policy(env, agent, cfg, done_add=False)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    agent.train = False
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)


def BipedalWalker_ddpg_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'BipedalWalker-v3'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        # save_path=r'D:\TMP\ddpg_BipedalWalker-v3_test_actor-v2.ckpt', # 1000次 291
        save_path=os.path.join(path_, "test_models" ,'DDPG_BipedalWalker-v3_test_actor-0.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[128, 64],
        # agent参数
        actor_lr=5e-5, # 3e-5,
        critic_lr=1e-3, # 5e-4
        gamma=0.99,
        # 训练参数
        num_episode=5000,
        sample_size=128,
        off_buffer_size=20480,
        off_minimal_size=2048,
        max_episode_rewards=900,
        max_episode_steps=300,
        # agent 其他参数
        DDPG_kwargs={
            'tau': 0.05, # soft update parameters
            'sigma': 0.5, # noise
            'action_bound': 1.0,
            'action_low': env.action_space.low[0],
            'action_high': env.action_space.high[0],
        }
    )
    agent = DDPG(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        DDPG_kwargs=cfg.DDPG_kwargs,
        device=cfg.device
    )
    agent.train = True
    # train_off_policy(env, agent, cfg, done_add=False)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    agent.train = False
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)



def carRacing_ddpg_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'CarRacing-v2'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'DDPG_CarRacing-v2_test_actor-0.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[128, 64],
        # agent参数
        actor_lr=5e-5, # 3e-5,
        critic_lr=1e-3, # 5e-4
        gamma=0.99,
        # 训练参数
        num_episode=100,
        sample_size=128,
        off_buffer_size=20480,
        off_minimal_size=2048,
        max_episode_rewards=900,
        max_episode_steps=300,
        # agent 其他参数
        DDPG_kwargs={
            'tau': 0.05, # soft update parameters
            'sigma': 0.5, # noise
            'sigma_exp_reduce_factor': 0.999,
            'action_low': env.action_space.low[0],
            'action_high': env.action_space.high[0],
        }
    )
    # 需要调整网络适应于图片输入
    agent = DDPG(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        DDPG_kwargs=cfg.DDPG_kwargs,
        device=cfg.device
    )
    agent.train = True
    train_off_policy(env, agent, cfg, done_add=False)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    agent.train = False
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)



if __name__ == '__main__':
    LunarLanderContinuous_ddpg_test()
    # BipedalWalker_ddpg_test()
    # carRacing_ddpg_test()

