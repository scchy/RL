
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


def BipedalWalkerHardcore_TD3_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'BipedalWalkerHardcore-v3'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'TD3_BipedalWalkerHardcore-v3_test_actor-3GPU.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        # 训练参数
        num_episode=1000,
        sample_size=256,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=int(1e6),
        off_minimal_size=4096,
        max_episode_rewards=1000,
        max_episode_steps=1000,
        # agent 其他参数
        TD3_kwargs={
            'action_low': env.action_space.low[0],
            'action_high': env.action_space.high[0],
            # soft update parameters
            'tau': 0.005, 
            # trick2: Delayed Policy Update
            'delay_freq': 1,
            # trick3: Target Policy Smoothing
            'policy_noise': 0.2,
            'policy_noise_clip': 0.5,
            # exploration noise
            'expl_noise': 0.25,
            # 探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t
            'expl_noise_exp_reduce_factor': 0.999,
            'off_minimal_size': 4096
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
    # agent.actor.load_state_dict(
    #     torch.load(os.path.join(path_, "test_models" ,'TD3_BipedalWalkerHardcore-v3_test_actor.ckpt'))
    # )

    agent.actor.load_state_dict(
        torch.load(os.path.join(path_, "test_models" ,'TD3_BipedalWalkerHardcore-v3_test_actor-3.ckpt'))
    )
    agent.train = True
    train_off_policy(env, agent, cfg, done_add=False, reward_func=reward_func)
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
        save_path=os.path.join(path_, "test_models" ,'TD3_CarRacing-v2_test2-3'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128], # 256
        critic_hidden_layers_dim=[128],
        # agent参数
        #    train_without_seed=True skip=5  out reward=-10
        # actor_lr=7.5e-5, # 1e-4,
        # critic_lr=1.5e-3, #2.5e-3, # 3e-3,
        #    train_without_seed=True skip=10 out reward=-10 policy+LayerNorm
        actor_lr=2.5e-4, #5.5e-5,
        critic_lr=1e-3, #7.5e-4,  

        gamma=0.99,
        # 训练参数
        num_episode=15000,
        sample_size=128,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=1024*100,  
        off_minimal_size=256,
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
            'expl_noise_exp_reduce_factor':  1 - 1e-4
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
    # train_1 = os.path.join(path_, "test_models" ,'TD3_CarRacing-v2_test2-2')
    # agent.load_model(train_1)
    # agent.train()
    # train_off_policy(env, agent, cfg, done_add=False, train_without_seed=True, wandb_flag=False, test_ep_freq=100)
    agent.load_model(cfg.save_path)
    agent.eval()
    # state, _ = env.reset()
    # state = torch.stack(state._frames).float().to(cfg.device)
    # act = agent.actor(state)
    # act.detach().cpu().numpy()[0].clip(agent.action_low, agent.action_high)
    env = gym.make(env_name, render_mode='human') # 
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(CarV2SkipFrame(env, skip=5)), 
            shape=84
        ), 
        num_stack=4
    )
    play(env, agent, cfg, episode_count=2)


def play1(env, cfg, episode_count=2):
    """
    对训练完成的QNet进行策略游戏
    """
    def random_action():
        return np.random.uniform(env.action_space.low, env.action_space.high)
    
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            env.render()
            a = random_action()
            n_state, reward, done, info1, info2 = env.step(a)
            # print(done, info1, info2)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (episode_reward >= 3 * cfg.max_episode_rewards) or (episode_cnt >= 3 * cfg.max_episode_steps):
                break


        print(f'Get reward {episode_reward}. Last {episode_cnt} times')


def test_env():
    env_name = 'CarRacing-v2'
    env_ = gym.make(env_name, render_mode=None)
    cfg = Config(
        env_, 
        # 环境参数
        seed=42,
        # 网络参数
        cnn_feature_dim=64,
        # agent参数
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        # 训练参数
        num_episode=10,
        sample_size=256,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=2048,
        off_minimal_size=512,
        max_episode_rewards=1000,
        max_episode_steps=4000
    )
    play1(gym.make(env_name, render_mode='human'), cfg, episode_count=2)


def InvertedPendulum_TD3_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'InvertedPendulum-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'TD3_InvertedPendulum-v4_test1.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        # 训练参数
        num_episode=1000,
        sample_size=128,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=int(1e6),
        off_minimal_size=512,
        max_episode_rewards=1000,
        max_episode_steps=1000,
        # agent 其他参数
        TD3_kwargs={
            'CNN_env_flag': 0,
            'action_low': env.action_space.low,
            'action_high': env.action_space.high,
            # soft update parameters
            'tau': 0.005, 
            # trick2: Delayed Policy Update
            'delay_freq': 1,
            # trick3: Target Policy Smoothing
            'policy_noise': 0.2,
            'policy_noise_clip': 0.5,
            # exploration noise
            'expl_noise': 0.5,
            # 探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t
            'expl_noise_exp_reduce_factor': 1 - 1e-4,
            'off_minimal_size': 4096
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
    # agent.train()
    # train_off_policy(env, agent, cfg, done_add=False, train_without_seed=True, wandb_flag=False, test_ep_freq=100)
    agent.load_model(cfg.save_path)
    agent.eval()
    play_env = gym.make(env_name, render_mode='human')
    play(play_env, agent, cfg, episode_count=2, render=True)



if __name__ == '__main__':
    # BipedalWalkerHardcore_TD3_test()
    # test_env()
    # CarRacing_TD3_test()
    InvertedPendulum_TD3_test()
    

