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
from RLAlgo.PPO import PPO
from RLAlgo.PPO2 import PPO2
from RLUtils import train_on_policy, play, Config, gym_env_desc


def r_func(r):
    # 不希望留在谷底
    return (r + 10.0 ) / 10.0

def r_InvertedPendulum(r):
    return (r - torch.mean(r) ) / torch.std(r) 


def ppo_InvertedPendulum_test():
    """
    policyNet: 
    valueNet: 
    """
    # A:[128, 64] C:[64, 32]
    env_name = 'InvertedPendulum-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO-InvertedPendulum-v4_test_actor-0.ckpt'),  
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 128],
        critic_hidden_layers_dim=[256, 256],
        # agent参数
        actor_lr=5e-4,
        critic_lr=5e-3,
        gamma=0.99,
        # 训练参数
        num_episode=18000,
        off_buffer_size=204800,
        max_episode_steps=260,
        # agent其他参数
        PPO_kwargs={
            'lmbda': 0.95,
            'eps': 0.2, # clip eps
            'k_epochs': 12,
            'sgd_batch_size': 512,
            'minibatch_size': 128,
            'actor_bound': env.action_space.high[0] # tanh (-1, 1)
        }
    )
    agent = PPO(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=r_InvertedPendulum
    )
    train_on_policy(env, agent, cfg, wandb_flag=False, step_lr_flag=True, step_lr_kwargs={'step_size': 3000, 'gamma': 0.9})
    agent.actor.load_state_dict(
        torch.load(cfg.save_path)
    )
    cfg.max_episode_steps = 300
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=3)



def ppo_test():
    """
    policyNet: 
    valueNet: 
    """
    # A:[128, 64] C:[64, 32]
    env_name = 'Pendulum-v1'
    # env_name = 'MountainCarContinuous-v0'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    cfg = Config(
        env, 
        # 环境参数
        save_path=r'D:\TMP\ppo_test_actor.ckpt', 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[64, 32],
        # agent参数
        actor_lr=1e-4,
        critic_lr=5e-3,
        gamma=0.9,
        # 训练参数
        num_episode=1200,
        off_buffer_size=204800,
        max_episode_steps=260,
        # agent其他参数
        PPO_kwargs={
            'lmbda': 0.9,
            'eps': 0.2, # clip eps
            'k_epochs': 10,
            'sgd_batch_size': 512,
            'minibatch_size': 128,
            'actor_nums': 3,
            'actor_bound': 2 # tanh (-1, 1)
        }
    )
    agent = PPO(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=r_func
    )
    train_on_policy(env, agent, cfg)
    agent.actor.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)


def HalfCheetah_v4_ppo_test():
    """
    policyNet: 
    valueNet: 
    """
    # A:[128, 64] C:[64, 32]
    env_name = 'HalfCheetah-v4'
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    env = gym.make(env_name, forward_reward_weight=2.0)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO_HalfCheetah-v4_test1'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        # agent参数
        actor_lr=5e-4,
        critic_lr=1e-3,
        gamma=0.98,
        # 训练参数
        num_episode=10000,
        off_buffer_size=102400, #204800,
        max_episode_steps=500,
        # agent其他参数
        PPO_kwargs={
            'lmbda': 0.9,
            'eps': 0.2, # clip eps
            'k_epochs': 10,
            'sgd_batch_size': 128,
            'minibatch_size': 12,
            'actor_nums': 3,
            'actor_bound': 1
        }
    )
    agent = PPO(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=r_func
    )
    # agent.train()
    # train_on_policy(env, agent, cfg, wandb_flag=False, train_without_seed=False, test_ep_freq=500)
    agent.load_model(cfg.save_path)
    agent.eval()
    env_ = gym.make(env_name, render_mode='human')
    play(env_, agent, cfg, episode_count=2, render=True)


# Hopper-v4
def Hopper_v4_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Hopper-v4'
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    # step1: 不限制unhealthy 但是对healthy_reward进行加权 step不能太多（噪声样本会变多） [v4_test2: max_episode_steps=160] dao zhi xuehuiqingdao
    env = gym.make(env_name, healthy_reward=1.5, terminate_when_unhealthy=False) # daoxia 
    # env = gym.make(env_name, healthy_reward=2.0, forward_reward_weight=1.0, 
    #                healthy_state_range=(-150, 150),      # default (-100, 100)
    #                healthy_z_range=(0.5, float('inf')),  # default (0.7, float("inf"))
    #                healthy_angle_range=(-0.45, 0.45),      # default (-0.2, 0.2)
    #                terminate_when_unhealthy=False)
    # step2: 限制unhealthy  对forward_reward_weight进行加权 [v4_test3: max_episode_steps=560] 
    # env = gym.make(env_name, healthy_reward=2.0, forward_reward_weight=1.0, 
    #                healthy_state_range=(-150, 150),      # default (-100, 100)
    #                healthy_z_range=(0.2, float('inf')),  # default (0.7, float("inf"))
    #                healthy_angle_range=(-0.45, 0.45),      # default (-0.2, 0.2)
    #                terminate_when_unhealthy=True)

    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO_Hopper-v4_test2'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[256, 256], # [200, 200],
        critic_hidden_layers_dim=[256, 256],
        # agent参数
        # step1
        actor_lr=2e-4, # 5e-4,
        critic_lr=7.5e-4,  # 1e-3
        # step2
        # actor_lr=1e-4, # 2.5e-4,
        # critic_lr=3.5e-4, # 5e-4, 
        gamma=0.98,
        # 训练参数
        num_episode=1000,
        off_buffer_size=10240,
        max_episode_steps=500,
        PPO_kwargs={
            'lmbda': 0.9,
            'eps': 0.2,
            'k_epochs': 8, 
            'sgd_batch_size': 256,
            'minibatch_size': 64, 
            'actor_nums': 3,
            'actor_bound': 1
        }
    )
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
    # # agent.load_model(os.path.join(path_, "test_models" ,'PPO_Hopper-v4_test2'))
    agent.train()
    train_on_policy(env, agent, cfg, wandb_flag=False, train_without_seed=True, test_ep_freq=1000, 
                    online_collect_nums=cfg.off_buffer_size,
                    test_episode_count=5)
    agent.load_model(cfg.save_path)
    agent.eval()
    env_ = gym.make(env_name) #, render_mode='human')
    play(env_, agent, cfg, episode_count=2, render=False) # render=True)  #  


if __name__ == '__main__':
    # ppo_InvertedPendulum_test()
    # HalfCheetah_v4_ppo_test()
    Hopper_v4_ppo2_test()
