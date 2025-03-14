import os
from os.path import dirname
import sys
import gymnasium as gym
import numpy as np
import cloudpickle
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
from RLAlgo.PPO2_old import PPO2 as PPO2_old
from RLAlgo.PPO2 import PPO2
from RLUtils import train_on_policy, random_play, play, Config, gym_env_desc, ppo2_train
from RLUtils import make_env, save_env


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
    agent = PPO2_old(
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
    info1: beta norm_flag=True, env_nums=128, off_buffer_size=max_episode_steps=80,actor_lr=4.5e-4,max_grad_norm=3.5 5000+ 斜着走
    info2: beta norm_flag=True, env_nums=128, off_buffer_size=max_episode_steps=80,actor_lr=2.5e-4,max_grad_norm=3.5 5000+
    info3: beta norm_flag=True, env_nums=64, off_buffer_size=max_episode_steps=64,actor_lr=6.5e-4,max_grad_norm=3.5,eps=0.2 5000+
    info4: norm oth same as info1
    """
    # [ Humanoid-v4 ](state: (376,),action: (17,)(连续 <-0.4 -> 0.4>))
    env_name = 'Humanoid-v4'
    num_envs = 128 #64
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'norm' # 'beta' 
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_Humanoid-v4-{norm_flag}-4'), 
        seed=202405,
        # 网络参数
        actor_hidden_layers_dim=[128, 128, 128],
        critic_hidden_layers_dim=[128, 128, 128],
        # agent参数
        actor_lr=4.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=3000, 
        off_buffer_size=80, # batch_size = off_buffer_size * num_env
        max_episode_steps=80,
        PPO_kwargs={
            'lmbda': 0.985, 
            'eps': 0.125,  
            'k_epochs': 3,
            'sgd_batch_size': 2048, # 1024, # 512,
            'minibatch_size': 1024,  # 512,  # 64,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': dist_type,
            'critic_coef': 1,
            'max_grad_norm': 3.5, # 45.5
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 3000
        }
    )
    cfg.test_max_episode_steps = 300
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)
    
    obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    env = make_env(env_name, obs_norm_trans_flag=norm_flag, render_mode='human')()
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    # env = make_env(env_name, obs_norm_trans_flag=norm_flag)()
    # cfg.max_episode_steps = 1020 
    play(env, agent, cfg, episode_count=3, play_without_seed=False, render=True)


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
    dist_type = 'beta' # beta norm
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_Pendulum-v1_{dist_type}'), 
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
            'dist_type': dist_type  ,
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
    agent.train()
    ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
                    train_without_seed=False, test_ep_freq=10000, 
                    online_collect_nums=cfg.off_buffer_size,
                    test_episode_count=10)
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_env(env_name)() #, render_mode='human')()
    play(env, agent, cfg, episode_count=6, play_without_seed=False, render=False)


# Hopper-v4
def Hopper_v4_ppo2_new_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Hopper-v4'
    num_envs = 4
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, exclude_current_positions_from_observation=True) for _ in range(num_envs)]
    )
    dist_type = 'beta' # beta norm
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_Hopper-v4_{dist_type}'), 
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
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=10000, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    # # render_mode='human'
    env = make_env(env_name, exclude_current_positions_from_observation=True, render_mode='human')()
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=True)


def Ant_v4_ppo2_test():
    env_name = 'Ant-v4'
    num_envs = 64
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_Ant-v4-{norm_flag}-1'), 
        seed=202405,
        # 网络参数
        actor_hidden_layers_dim=[128, 128],
        critic_hidden_layers_dim=[128, 128],
        # agent参数
        actor_lr=5.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=2000, 
        off_buffer_size=80,
        max_episode_steps=80,
        PPO_kwargs={
            'lmbda': 0.985, 
            'eps': 0.125,  
            'k_epochs': 3,
            'sgd_batch_size': 2048,
            'minibatch_size': 1024,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': dist_type,
            'critic_coef': 1,
            'max_grad_norm': 3.5, # 45.5
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 3000
        }
    )
    cfg.test_max_episode_steps = 300
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)

    obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    env = make_env(env_name, obs_norm_trans_flag=norm_flag, render_mode='human')()
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    # env = make_env(env_name, obs_norm_trans_flag=norm_flag)()
    play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True)


def Swimmer_v4_ppo2_test():
    """
    The episode truncates when the episode length is greater than 1000.
    
    tanh -> relu
    max_grad_norm 0.5 -> 1.9 -> 2.5 -> 5.5 
    dim 125 -> 200
    rl 3.5e-4 -> 1.5e-4  
    off_buffer_size 64 -> 100  & batch_reset_env=True
    fix test_env bug
    -- next try X 
    off_buffer_size -> 32
    num_envs -> 64 
    batch_reset_env=True
    """
    env_name = 'Swimmer-v4'
    num_envs = 12
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True  
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'norm'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_{env_name}-{norm_flag}-1'), 
        seed=202408,
        # 网络参数
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        actor_lr=3.5e-4,
        gamma=0.99,
        # 训练参数
        num_episode=3500, 
        off_buffer_size=80, # total-exp: off_buffer_size * num_envs
        max_episode_steps=80,
        PPO_kwargs={
            'lmbda': 0.98, 
            'eps': 0.165,  
            'k_epochs': 3,
            'sgd_batch_size':  256,
            'minibatch_size':  128, 
            'action_space': envs.single_action_space,
            'act_type': 'tanh', #
            'dist_type': dist_type,
            'critic_coef': 1.5,
            'max_grad_norm': 5.5, # 2.5, #1.9, 
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 5000
        }
    )
    cfg.test_max_episode_steps = 100  
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=False, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10,
    #                 batch_reset_env=False, # True, # True
    #                 add_max_step_reward_flag=True,
    #                 play_func='ppo2_play'
    #                 )
    # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)
        obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    
    env = make_env(env_name, obs_norm_trans_flag=norm_flag)()#, render_mode='human')()
    if norm_flag:
        env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
        env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
        env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    cfg.test_max_episode_steps = 400 # 62
    play(env, agent, cfg, episode_count=3, play_without_seed=False, render=False)



def Walker2d_v4_ppo2_test():
    # jump & another leg to be a tail 
    env_name = 'Walker2d-v4'
    num_envs = 64
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_{env_name}-{norm_flag}-1'), 
        seed=202405,
        # 网络参数
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        actor_lr=3.5e-4,  
        gamma=0.99,
        # 训练参数
        num_episode=2600, 
        off_buffer_size=80,
        max_episode_steps=80,
        PPO_kwargs={
            'lmbda': 0.985,  
            'eps': 0.165,
            'k_epochs': 3,
            'sgd_batch_size': 2048,
            'minibatch_size': 1024,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': dist_type,
            'critic_coef': 1,
            'max_grad_norm': 3.5, # 45.5
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 3000
        }
    )
    cfg.test_max_episode_steps = 300
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)

    obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    env = make_env(env_name, obs_norm_trans_flag=norm_flag, render_mode='human')()
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    # env = make_env(env_name, obs_norm_trans_flag=norm_flag)()
    play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True)


def HumanoidStandup_v4_ppo2_test():
    # jump & another leg to be a tail 
    env_name = 'HumanoidStandup-v4'
    num_envs = 128
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_{env_name}-{norm_flag}-1'), 
        seed=202405,
        # 网络参数
        actor_hidden_layers_dim=[256, 256, 256],
        critic_hidden_layers_dim=[256, 256, 256],
        # agent参数
        actor_lr=3.5e-4,  # 4.5e-4
        gamma=0.99,
        # 训练参数
        num_episode=3500, 
        off_buffer_size=80,
        max_episode_steps=80,
        PPO_kwargs={
            'lmbda': 0.985,  
            'eps': 0.225, # 0.2
            'k_epochs': 3,
            'sgd_batch_size': 2048,
            'minibatch_size': 1024,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': dist_type,
            'critic_coef': 1,
            'max_grad_norm': 3.5, # 45.5
            'clip_vloss': True,
            # 'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 3000
        }
    )
    cfg.test_max_episode_steps = 300
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=lambda r: r/1000.0
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)

    obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    env = make_env(env_name, obs_norm_trans_flag=norm_flag, render_mode='human')()
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    # env = make_env(env_name, obs_norm_trans_flag=norm_flag)()
    play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True)


# Reacher-v4
def Reacher_v4_ppo2_test():
    # jump & another leg to be a tail 
    env_name = 'Reacher-v4'
    num_envs = 64
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    norm_flag = True
    reward_flag = False
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, obs_norm_trans_flag=norm_flag, reward_norm_trans_flag=reward_flag) for _ in range(num_envs)]
    )
    dist_type = 'beta'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO_{env_name}-{norm_flag}-1'), 
        seed=202410,
        # 网络参数
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        # agent参数
        actor_lr=4.5e-4,
        gamma=0.99,
        # 训练参数
        num_episode=3000, 
        off_buffer_size=100,
        max_episode_steps=100,
        PPO_kwargs={
            'lmbda': 0.95,  
            'eps': 0.2,
            'k_epochs': 3,
            'sgd_batch_size': 2048,
            'minibatch_size': 1024,
            'action_space': envs.single_action_space,
            'act_type': 'tanh',
            'dist_type': dist_type,
            'critic_coef': 1,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5, # 45.5
            'clip_vloss': True,
            'min_adv_norm': True,

            'anneal_lr': False, # not work
            'num_episode': 3000
        }
    )
    cfg.test_max_episode_steps = 300
    cfg.num_envs = num_envs
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"reward_flag={reward_flag},norm_flag={norm_flag},actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
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
        reward_func=None
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name}",
    #                 train_without_seed=True,  # False
    #                 test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10)
    # # save norm env
    # save_env(envs.envs[0], os.path.join(cfg.save_path, 'norm_env.pkl'))
    # print(agent.grad_collector.describe())
    # agent.grad_collector.dump(cfg.save_path + '.npy')
    agent.load_model(cfg.save_path)
    agent.eval()
    
    with open(os.path.join(cfg.save_path, 'norm_env.pkl'), 'rb') as f:
        env = cloudpickle.load(f)

    obs_rms = env.get_wrapper_attr('env').get_wrapper_attr("obs_rms")
    env = make_env(env_name, obs_norm_trans_flag=norm_flag, render_mode='human')()
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").mean = obs_rms.mean
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").var = obs_rms.var
    env.get_wrapper_attr('env').get_wrapper_attr("obs_rms").count = obs_rms.count
    # env = make_env(env_name, obs_norm_trans_flag=norm_flag)()
    play(env, agent, cfg, episode_count=10, play_without_seed=True, render=True)


if __name__ == '__main__':
    # ppo_InvertedPendulum_test()
    # HalfCheetah_v4_ppo_test()
    # Hopper_v4_ppo2_test()
    # rd_hopper()
    # https://wandb.ai/296294812/PPO2-Pendulum-v1?nw=nwuser296294812
    # Pendulum_v1_ppo2_test() # 2024-04-{16, 18} norm & beta
    # https://wandb.ai/296294812/PPO2-Hopper-v4?nw=nwuser296294812
    # Hopper_v4_ppo2_new_test() # 2024-04-19 
    # Humanoid_v4_ppo2_test() # 2024-05-27 - beta / 2024-05-30 - norm 
    # Ant_v4_ppo2_test() # 2024-05-28
    # Swimmer_v4_ppo2_test() # 2024-08-27
    # Walker2d_v4_ppo2_test() # 2024-07-01 
    # HumanoidStandup_v4_ppo2_test() # 2024-07-05
    Reacher_v4_ppo2_test() # 2024-10-11

