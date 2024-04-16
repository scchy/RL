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
from RLAlgo.PPO import PPO
from RLAlgo.PPO2 import PPO2
from RLUtils import train_on_policy, random_play, play, Config, gym_env_desc, ppo2_train
from RLUtils import make_env
from RLUtils.env_wrapper import FrameStack, baseSkipFrame, GrayScaleObservation, ResizeObservation


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
    env = gym.make(
        env_name, 
        exclude_current_positions_from_observation=True,
        # healthy_reward=0
    )
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO_Hopper-v4_test2'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[256, 256, 256],
        critic_hidden_layers_dim=[256, 256, 256],
        # agent参数
        actor_lr=1.5e-4,
        critic_lr=5.5e-4,
        gamma=0.99,
        # 训练参数
        num_episode=12500,
        off_buffer_size=512,
        off_minimal_size=510,
        max_episode_steps=500,
        PPO_kwargs={
            'lmbda': 0.9,
            'eps': 0.25,
            'k_epochs': 4, 
            'sgd_batch_size': 128,
            'minibatch_size': 12, 
            'actor_bound': 1,
            'dist_type': 'beta'
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
    # agent.train()
    # train_on_policy(env, agent, cfg, wandb_flag=False, train_without_seed=True, test_ep_freq=1000, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=5)
    agent.load_model(cfg.save_path)
    agent.eval()
    env_ = gym.make(env_name, 
                    exclude_current_positions_from_observation=True,
                    # render_mode='human'
                    ) # , render_mode='human'
    play(env_, agent, cfg, episode_count=3, play_without_seed=True, render=False)


def rd_hopper():
    env_name = 'Hopper-v4'
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    env = gym.make(
        env_name, 
        render_mode='human',
        exclude_current_positions_from_observation=True,
    )
    random_play(env, episode_count=10)



def Humanoid_v4_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    # [ Humanoid-v4 ](state: (376,),action: (17,)(连续 <-0.4 -> 0.4>))
    env_name = 'Humanoid-v4'
    num_envs = 10
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    env = gym.make(env_name)  
    # env = TransformObservation(
    #         NormalizeObservation(ClipAction(env)), lambda obs: np.clip(obs, -10, 10)
    # )
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=True) for _ in range(num_envs)]
    )
    # env = TransformReward(NormalizeReward(env, gamma=0.99), lambda reward: np.clip(reward, -10, 10))
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO_Humanoid-v4_test-2'), 
        seed=202403,
        # 网络参数
        actor_hidden_layers_dim=[128, 128, 128],
        critic_hidden_layers_dim=[128, 128, 128],
        # agent参数
        actor_lr=3.0e-3, #3.0e-4,
        critic_lr=3.0e-3,
        gamma=0.99,
        # 训练参数
        num_episode=500, #350000//4096,
        off_buffer_size=2048, # 2048
        max_episode_steps=64,
        PPO_kwargs={
            'lmbda': 0.95,
            'eps': 0.2,
            'k_epochs': 2,  # update_epochs
            'sgd_batch_size': 1024, # 128
            'minibatch_size': 512, #48, # 96
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': 'norm',
            'critic_coef': 1.3,
            'act_attention_flag': False, # relation of different actions
            'max_grad_norm': 3.5,
            'clip_vloss': True,

            'anneal_lr': True,
            'num_episode': 350000,
            'off_buffer_size': 2048
        }
    )
    cfg.state_dim = envs.single_observation_space.shape[0]
    cfg.action_dim = envs.single_action_space.shape[0]
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
    agent.train()
    ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name="PPO2",
                    train_without_seed=True, test_ep_freq=40000, 
                    online_collect_nums=cfg.off_buffer_size,
                    test_episode_count=5)
    # {'P25': 0.08523080311715603, 'P50': 7.071380615234375, 'P75': 34.25641632080078, 
    # 'P95': 60.77569007873534, 'P99': 108.51971694946333}
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    env = gym.make(env_name)#, render_mode='human')  
    # env = TransformObservation(
    #         NormalizeObservation(ClipAction(env)), lambda obs: np.clip(obs, -10, 10)
    # )
    # env = TransformReward(NormalizeReward(env, gamma=0.99), lambda reward: np.clip(reward, -10, 10))
    play(env, agent, cfg, episode_count=6, play_without_seed=False, render=False)


def Pendulum_v1_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Pendulum-v1'
    num_envs = 4
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name) for _ in range(num_envs)]
    )
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'PPO2_Pendulum-v1_test-2'), 
        seed=202403,
        # 网络参数
        actor_hidden_layers_dim=[128, 128],
        critic_hidden_layers_dim=[128, 128],
        # agent参数
        actor_lr=5.5e-4,
        gamma=0.99,
        # 训练参数
        num_episode=500,
        off_buffer_size=1024,
        max_episode_steps=200,
        PPO_kwargs={
            'lmbda': 0.95,
            'eps': 0.2,
            'k_epochs': 4,  # update_epochs
            'sgd_batch_size': 64,
            'minibatch_size': 48,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': 'norm',
            'critic_coef': 1.0,
            'max_grad_norm': 0.5, # close == 0
            'clip_vloss': True,

            'anneal_lr': False,
            'num_episode': 1200,
            'off_buffer_size': 1024
        }
    )
    cfg.state_dim = envs.single_observation_space.shape[0]
    cfg.action_dim = envs.single_action_space.shape[0]
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
        reward_func=lambda r: (r + 10.0 ) / 10.0
    )
    # agent.train()
    # ppo2_train(env_name, envs, agent, cfg, wandb_flag=False, wandb_project_name="PPO2",
    #                 train_without_seed=False, test_ep_freq=10000, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_env(env_name, render_mode='human')()
    play(env, agent, cfg, episode_count=6, play_without_seed=False, render=False)



if __name__ == '__main__':
    # ppo_InvertedPendulum_test()
    # HalfCheetah_v4_ppo_test()
    # Hopper_v4_ppo2_test()
    # rd_hopper()
    # Humanoid_v4_ppo2_test()
    Pendulum_v1_ppo2_test()
