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
from RLAlgo.PPO2 import PPO2, cnnICMPPO2
from RLUtils import play, Config, gym_env_desc, ppo2_train, ppo2_play
from RLUtils import make_atari_env, make_envpool_atria, spSyncVectorEnv


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
            # 'mini_adv_norm': True,

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
    # env = make_atari_env(env_name, clip_reward=True)()
    cfg.max_episode_steps = 1620 
    play(env, agent, cfg, episode_count=3, play_without_seed=False, render=True, ppo_train=True)



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
            # 'mini_adv_norm': True,

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
            # 'mini_adv_norm': True,

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


# Breakout
def Breakout_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/Breakout-v5' 
    env_name_str = env_name.replace('/', '-')
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    num_envs = 12 #16 # 20 # 10
    episod_life = True
    clip_reward = True
    resize_inner_area = True # True
    env_pool_flag = False # True
    seed = 202404
    if env_pool_flag:
        envs = make_envpool_atria(env_name.split('/')[-1], num_envs, seed=seed, episodic_life=episod_life, reward_clip=clip_reward, max_no_reward_count=200)
        ply_env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=200)
    else: 
        envs = spSyncVectorEnv(
            [make_atari_env(env_name, skip=4, episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                            max_no_reward_count=120, resize_inner_area=resize_inner_area) for _ in range(num_envs)],
            random_reset=False,
            seed=202404
        )
        ply_env = None 
        # make_atari_env(env_name, skip=4, episod_life=episod_life, clip_reward=False, ppo_train=True, max_no_reward_count=120, resize_inner_area=resize_inner_area)()
    dist_type = 'norm'
    # beta [1:01:35<00:00,  3.70s/it, lastMeanRewards=28.90, BEST=34.30, bestTestReward=25.70] Get reward 18.0.
    cfg = Config(
        ply_env if env_pool_flag else envs, 
        # 环境参数
        # 1- PPO2__AtariEnv instance__20241030__0204  reward=407
        # 2- PPO2__AtariEnv instance__20241029__2217  reward=416
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-2'),  
        seed=202404,
        num_envs=num_envs,
        episod_life=episod_life,
        clip_reward=clip_reward,
        resize_inner_area=resize_inner_area,
        env_pool_flag=env_pool_flag,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[512, 256], 
        critic_hidden_layers_dim=[512, 128], 
        # agent参数
        actor_lr=4.5e-4,  # 5.5e-4,  # low lr - slow  # high 1e-3 not work 
        gamma=0.99,
        # 训练参数
        num_episode=3600,  
        off_buffer_size=128, # 128, # 128, #120, # batch_size = off_buffer_size * num_env
        max_episode_steps=128, 
        PPO_kwargs={
            'cnn_flag': True,
            'clean_rl_cnn': True,
            'share_cnn_flag': True,
            'continue_action_flag': False,

            'lmbda': 0.95,
            'eps':  0.165,  # 0.165
            'k_epochs': 4,  #  update_epochs
            'sgd_batch_size': 512,  
            'minibatch_size': 256, 
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.0, # 1.0
            'ent_coef': 0.05, 
            # 0.015 & 256+128batch 陡降-回升慢  
            # 0.025 & 256          陡降回升-311
            # Best 0.05 & 256 354-383 
            # 0.05 & 256+128
            # 0.1 & 256            提升过于平缓
            'max_grad_norm': 0.5,  
            'clip_vloss': True,
            'mini_adv_norm': True,

            'anneal_lr': False,
            'num_episode': 3600,
            # "avg_pooling": False, 
            # "max_pooling": True
        }
    )
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
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}-NEW",
    #                 train_without_seed=False, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10, 
    #                 add_max_step_reward_flag=False,
    #                 play_func='ppo2_play',
    #                 ply_env=ply_env
    # )
    # print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    # env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=200)
    env = make_atari_env(env_name, skip=4, episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                        max_no_reward_count=120, resize_inner_area=resize_inner_area, render_mode='human')()

    cfg.max_episode_steps = 1620 
    ppo2_play(env, agent, cfg, episode_count=2, play_without_seed=False, render=True, ppo_train=True)



def DoubleDunk_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/DoubleDunk-v5' 
    env_name_str = env_name.replace('/', '-')
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    num_envs = 12 
    episod_life = True
    clip_reward = False
    resize_inner_area = True  
    env_pool_flag = False  
    fire_flag = False
    gray_flag = True
    start_skip = None
    seed = 202502
    max_no_reward_count = 1088 # 888   
    stack_num = 10
    clear_ball_reward = 0.01 # step1 & step2-try1
    shape = 96 # 124  bestTest -> -7.43
    if env_pool_flag:
        envs = make_envpool_atria(env_name.split('/')[-1], num_envs, seed=seed, episodic_life=episod_life, reward_clip=clip_reward, max_no_reward_count=max_no_reward_count)
        ply_env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=max_no_reward_count)
    else: 
        envs = spSyncVectorEnv(
            [make_atari_env(env_name, skip=1, start_skip=start_skip, cut_slices=[[30, 190]], episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                            max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area,
                            fire_flag=fire_flag, gray_flag=gray_flag, stack_num=stack_num, shape=shape, 
                            double_dunk=True, double_dunk_clear_ball_reward=clear_ball_reward
                            ) for _ in range(num_envs)],
            random_reset=True,
            seed=seed
        )
        ply_env = None 
    dist_type = 'norm'
    cfg = Config(
        ply_env if env_pool_flag else envs, 
        # 环境参数
        # -2: https://wandb.ai/296294812/PPO2-ALE-DoubleDunk-v5-F/runs/qs8ltiws/workspace?nw=nwuser296294812
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-1'),   #-2   lastMeanRewards=-5.20, BEST=1.80, bestTestReward=-0.67
        seed=seed,
        add_entroy_bonus_coef=0, #0.002,
        num_envs=num_envs,
        stack_num=stack_num,
        fire_flag=fire_flag,
        episod_life=episod_life,
        clip_reward=clip_reward,
        resize_inner_area=resize_inner_area,
        env_pool_flag=env_pool_flag,
        max_no_reward_count=max_no_reward_count,
        start_skip=start_skip,
        clear_ball_reward=clear_ball_reward,
        shape=shape,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[512], # , 256, 128],
        critic_hidden_layers_dim=[512, 256, 128],
        # agent参数
        actor_lr=1.5e-4, # 4.5e-4 learn shot  & clear ball += 0.01
        # 1.5e-4, # step2-try1 lastMeanRewards=0.00, BEST=1.20, bestTestReward=0.29] 
        gamma=0.99,
        # 训练参数
        num_episode=888,  # 1288 learn shot  & clear ball += 0.01
        off_buffer_size=360, # 256 # 360  on policy 见到更多当前策略的表现
        max_episode_steps=360, 
        PPO_kwargs={
            'cnn_flag': True,
            'clean_rl_cnn': True,
            'share_cnn_flag': True,
            'stack_num': stack_num,
            'continue_action_flag': False,
            'large_cnn': True,        
            'grey_flag': gray_flag,

            'lmbda': 0.95,
            'eps': 0.175,  # 0.175, learn shot
            'k_epochs': 3, # 3, 
            'sgd_batch_size': 1024,  
            'minibatch_size': 512, 
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.5,  
            'ent_coef': 0.0125,  # 0.0125 learn shot & clear ball += 0.01
            'max_grad_norm': 1.5,  #  learn shot
            'clip_vloss': True,
            'mini_adv_norm': False,

            'anneal_lr': False,
            'num_episode': 888,
        }
    )
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
        reward_func=lambda x: x * 2.0 # learn to shoot
    )
    # 'PPO2_{env_name_str}-2'  lastMeanRewards=-5.20, BEST=1.80, bestTestReward=-0.67
    # step2  clear_ball_reward=0.01  &  reduce lr -> 1.5e-4
    # agent.load_model(cfg.save_path.replace('-1', '-2'))
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}-F",
    #                 train_without_seed=True, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10, 
    #                 add_max_step_reward_flag=False,
    #                 play_func='ppo2_play',
    #                 ply_env=ply_env,
    #                 add_entroy_bonus=False,
    #                 add_entroy_bonus_coef=cfg.add_entroy_bonus_coef
    # )
    # print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    # env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=200)
    env = make_atari_env(env_name, skip=1, start_skip=start_skip, cut_slices=[[30, 190]], episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                         fire_flag=fire_flag, gray_flag=gray_flag, max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area, stack_num=stack_num, 
                         shape=shape, double_dunk=True, double_dunk_clear_ball_reward=clear_ball_reward
                        , render_mode='human')()
                        # )() 

    cfg.max_episode_steps = 1620 
    ppo2_play(env, agent, cfg, episode_count=3, play_without_seed=True, render=True, ppo_train=True)



def Bowling_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/Bowling-v5' 
    env_name_str = env_name.replace('/', '-')
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    num_envs = 24
    episod_life = True
    clip_reward = False
    resize_inner_area = True  
    gray_flag = True
    start_skip = None
    seed = 202503
    max_no_reward_count = 688 
    stack_num = 8
    shape = 84
    envs = spSyncVectorEnv(
        [make_atari_env(env_name, skip=2, start_skip=start_skip, cut_slices=None, episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                        max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area,
                        fire_flag=True, gray_flag=gray_flag, stack_num=stack_num, shape=shape) for _ in range(num_envs)],
        random_reset=True,
        seed=seed
    )
    dist_type = 'norm'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-2'),  # 
        seed=seed,
        num_envs=num_envs,
        stack_num=stack_num,
        episod_life=episod_life,
        clip_reward=clip_reward,
        resize_inner_area=resize_inner_area,
        max_no_reward_count=max_no_reward_count,
        start_skip=start_skip,
        shape=shape,
        actor_hidden_layers_dim=[1024, 512, 256],
        critic_hidden_layers_dim=[1024, 256, 128],
        # agent参数
        actor_lr=2.5e-4, # 2.5e-4,
        gamma=0.99,
        # 训练参数 
        num_episode=1688,  
        off_buffer_size=312,  # 312
        max_episode_steps=312, 
        PPO_kwargs={
            'cnn_flag': True,
            'clean_rl_cnn': True,
            'share_cnn_flag': True,
            'stack_num': stack_num,
            'continue_action_flag': False,
            'large_cnn': False,        
            'grey_flag': gray_flag,

            'lmbda': 0.95,
            'eps': 0.2, # 0.2
            'k_epochs': 3,  
            'sgd_batch_size': 512,
            'minibatch_size':  256,  
            'mini_adv_norm': False,
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.5,  
            'ent_coef': 0.01,  # 0.01, 
            'max_grad_norm': 0.5,
            'clip_vloss': True,

            'anneal_lr': False,
            'num_episode': 1288,
        }
    )
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
        reward_func=None,  
    )
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}",
    #                 train_without_seed=True, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10, 
    #                 add_max_step_reward_flag=False,
    #                 play_func='ppo2_play',
    #                 ply_env=None
    # )
    # print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    # env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=200)
    env = make_atari_env(env_name, skip=2, start_skip=start_skip, cut_slices=None, #[[100, 175]], 
                         episod_life=episod_life, clip_reward=clip_reward, 
                         ppo_train=True, fire_flag=True, gray_flag=gray_flag,
                        max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area, stack_num=stack_num, shape=shape
                        , render_mode='human')()
                        # )() 

    cfg.max_episode_steps = 1620 
    ppo2_play(env, agent, cfg, episode_count=3, play_without_seed=True, render=True, ppo_train=True)


def Galaxian_v5_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/Galaxian-v5' 
    env_name_str = env_name.replace('/', '-')
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    num_envs = 24
    episod_life = True
    clip_reward = True
    resize_inner_area = True  
    gray_flag = True
    start_skip = None
    seed = 202504
    max_no_reward_count = 688 
    stack_num = 8
    shape = 84
    envs = spSyncVectorEnv(
        [make_atari_env(env_name, skip=4, start_skip=start_skip, cut_slices=None, episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                        max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area,
                        fire_flag=True, gray_flag=gray_flag, stack_num=stack_num, shape=shape) for _ in range(num_envs)],
        random_reset=True,
        seed=seed
    )
    dist_type = 'norm'
    cfg = Config(
        envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'PPO2_{env_name_str}-1'),  # -2  skip=2
        seed=seed,
        num_envs=num_envs,
        stack_num=stack_num,
        episod_life=episod_life,
        clip_reward=clip_reward,
        resize_inner_area=resize_inner_area,
        max_no_reward_count=max_no_reward_count,
        start_skip=start_skip,
        shape=shape,
        actor_hidden_layers_dim=[1024],
        critic_hidden_layers_dim=[1024, 256, 128],
        # agent参数
        actor_lr=2.5e-4, 
        gamma=0.99,
        # 训练参数 
        num_episode=1088,  
        off_buffer_size=240,  
        max_episode_steps=240, 
        PPO_kwargs={
            'cnn_flag': True,
            'clean_rl_cnn': True,
            'share_cnn_flag': True,
            'stack_num': stack_num,
            'continue_action_flag': False,
            'large_cnn': False,        
            'grey_flag': gray_flag,

            'lmbda': 0.95,
            'eps': 0.185,
            'k_epochs': 3,  
            'sgd_batch_size': 1024,
            'minibatch_size':  512,  
            'mini_adv_norm': True,
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.2,  
            'ent_coef': 0.01,    
            'max_grad_norm': 0.5,
            'clip_vloss': True,

            'anneal_lr': False,
            'num_episode': 1288,
        }
    )
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
        reward_func=None,  
    )
    # agent.load_model(cfg.save_path) # continue learning
    # agent.train()
    # ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"PPO2-{env_name_str}",
    #                 train_without_seed=True, test_ep_freq=cfg.off_buffer_size * 10, 
    #                 online_collect_nums=cfg.off_buffer_size,
    #                 test_episode_count=10, 
    #                 add_max_step_reward_flag=False,
    #                 play_func='ppo2_play',
    #                 ply_env=None
    # )
    # print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    env = make_atari_env(env_name, skip=4, start_skip=start_skip, cut_slices=None, 
                         episod_life=episod_life, clip_reward=clip_reward, 
                         ppo_train=True, fire_flag=True, gray_flag=gray_flag,
                        max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area, stack_num=stack_num, shape=shape
                        , render_mode='human')()
                        # )() 

    cfg.max_episode_steps = 1620 
    ppo2_play(env, agent, cfg, episode_count=3, play_without_seed=True, render=True, ppo_train=True)



def DoubleDunk_v5_ICM_ppo2_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'ALE/DoubleDunk-v5' 
    env_name_str = env_name.replace('/', '-')
    gym_env_desc(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    num_envs = 12 
    episod_life = True
    clip_reward = False
    resize_inner_area = True  
    env_pool_flag = False  
    fire_flag = False
    gray_flag = True
    start_skip = None
    seed = 202502
    max_no_reward_count = 1088 # 888   
    stack_num = 10
    clear_ball_reward = 0.01 # step1 & step2-try1
    shape = 96  
    if env_pool_flag:
        envs = make_envpool_atria(env_name.split('/')[-1], num_envs, seed=seed, episodic_life=episod_life, reward_clip=clip_reward, max_no_reward_count=max_no_reward_count)
        ply_env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=max_no_reward_count)
    else: 
        envs = spSyncVectorEnv(
            [make_atari_env(env_name, skip=1, start_skip=start_skip, cut_slices=[[30, 190]], episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                            max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area,
                            fire_flag=fire_flag, gray_flag=gray_flag, stack_num=stack_num, shape=shape, 
                            double_dunk=True, double_dunk_clear_ball_reward=clear_ball_reward
                            ) for _ in range(num_envs)],
            random_reset=True,
            seed=seed
        )
        ply_env = None 
    dist_type = 'norm'
    cfg = Config(
        ply_env if env_pool_flag else envs, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,f'ICM_PPO2_{env_name_str}-1'),  
        seed=seed,
        add_entroy_bonus_coef=0,  
        num_envs=num_envs,
        stack_num=stack_num,
        fire_flag=fire_flag,
        episod_life=episod_life,
        clip_reward=clip_reward,
        resize_inner_area=resize_inner_area,
        env_pool_flag=env_pool_flag,
        max_no_reward_count=max_no_reward_count,
        start_skip=start_skip,
        clear_ball_reward=clear_ball_reward,
        shape=shape,
        # 网络参数 Atria-CNN + MLP
        actor_hidden_layers_dim=[512], 
        critic_hidden_layers_dim=[512, 256, 128],
        # agent参数
        actor_lr=4.5e-4, 
        gamma=0.99,
        # 训练参数
        num_episode=1688,
        off_buffer_size=360,
        max_episode_steps=360, 
        PPO_kwargs={
            'cnn_flag': True,
            'clean_rl_cnn': True,
            'share_cnn_flag': True,
            'stack_num': stack_num,
            'continue_action_flag': False,
            'large_cnn': True,        
            'grey_flag': gray_flag,

            'lmbda': 0.95,
            'eps': 0.175,  # 0.175, learn shot
            'k_epochs': 3, # 3, 
            'sgd_batch_size': 1024,  
            'minibatch_size': 512, 
            'act_type': 'relu',
            'dist_type': dist_type,
            'critic_coef': 1.5,  
            'ent_coef': 0.0125,
            'max_grad_norm': 1.5,  #  learn shot
            'clip_vloss': True,
            'mini_adv_norm': False,

            'anneal_lr': True,
            'num_episode': 1088,
            # ICM
            "icm_epochs": 1,
            "icm_batch_size": 1024,
            'icm_minibatch_size': 512, 
            "icm_intr_reward_strength": 0.0125,
            "icm_max_grad_norm": 1.5
        }
    )
    minibatch_size = cfg.PPO_kwargs['minibatch_size']
    max_grad_norm = cfg.PPO_kwargs['max_grad_norm']
    cfg.trail_desc = f"actor_lr={cfg.actor_lr},minibatch_size={minibatch_size},max_grad_norm={max_grad_norm},hidden_layers={cfg.actor_hidden_layers_dim}",
    agent = cnnICMPPO2(
    # agent = PPO2(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=lambda x: x * 2.0 # learn to shoot
    )
    agent.train()
    ppo2_train(envs, agent, cfg, wandb_flag=True, wandb_project_name=f"ICM_PPO2-{env_name_str}",
                    train_without_seed=True, test_ep_freq=cfg.off_buffer_size * 10, 
                    online_collect_nums=cfg.off_buffer_size,
                    test_episode_count=10, 
                    add_max_step_reward_flag=False,
                    play_func='ppo2_play',
                    ply_env=ply_env,
                    add_entroy_bonus=False,
                    add_entroy_bonus_coef=cfg.add_entroy_bonus_coef
    )
    print(agent.grad_collector.describe())
    agent.load_model(cfg.save_path)
    agent.eval()
    # env = make_envpool_atria(env_name.split('/')[-1], 1, seed=seed, episodic_life=False, reward_clip=False, max_no_reward_count=200)
    env = make_atari_env(env_name, skip=1, start_skip=start_skip, cut_slices=[[30, 190]], episod_life=episod_life, clip_reward=clip_reward, ppo_train=True, 
                         fire_flag=fire_flag, gray_flag=gray_flag, max_no_reward_count=max_no_reward_count, resize_inner_area=resize_inner_area, stack_num=stack_num, 
                         shape=shape, double_dunk=True, double_dunk_clear_ball_reward=clear_ball_reward
                        # , render_mode='human')()
                        )() 

    cfg.max_episode_steps = 1620 
    ppo2_play(env, agent, cfg, episode_count=3, play_without_seed=True, render=False, ppo_train=True)





if __name__ == '__main__':
    # DemonAttack_v5_ppo2_test() # 2024-04-25
    # AirRaid_v5_ppo2_test()
    # Alien_v5_ppo2_test()
    # Breakout_v5_ppo2_test() # 2024-10-30 
    # todo:  DoubleDunk_v5 & Bowling_v5 gif
    # DoubleDunk_v5_ppo2_test() # 2025-04-25 
    # Bowling_v5_ppo2_test() # 2025-03-30
    # Galaxian_v5_ppo2_test() # 2025-04-30
    DoubleDunk_v5_ICM_ppo2_test()

