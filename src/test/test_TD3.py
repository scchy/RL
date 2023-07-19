
import os
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.TD3 import TD3
from RLUtils import train_off_policy, play, Config, gym_env_desc
import numpy as np


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
    """
    env_name = 'CarRacing-v2'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        save_path=os.path.join(path_, "test_models" ,'TD3_CarRacing-v2_test'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[200],
        critic_hidden_layers_dim=[200],
        # agent参数
        actor_lr=1e-4,
        critic_lr=2e-3,
        gamma=0.99,
        # 训练参数
        num_episode=20,
        # num_episode=500,
        sample_size=256,
        # 环境复杂多变，需要保存多一些buffer
        off_buffer_size=2048*2,
        off_minimal_size=1024,
        max_episode_rewards=1000,
        max_episode_steps=150,
        # max_episode_steps=800,
        # agent 其他参数
        TD3_kwargs={
            'off_minimal_size': 1024,
            'CNN_env_flag': 1,
            'pic_shape': env.observation_space.shape,
            'action_low': env.action_space.low,
            'action_high': env.action_space.high,
            # soft update parameters
            'tau': 0.03, 
            # trick2: Delayed Policy Update
            'delay_freq': 1,
            # trick3: Target Policy Smoothing
            'policy_noise': 0.2,
            'policy_noise_clip': 0.5,
            # exploration noise
            'expl_noise': 0.25,
            # 探索的 noise 指数系数率减少 noise = expl_noise * expl_noise_exp_reduce_factor^t
            'expl_noise_exp_reduce_factor': 0.9999
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
    # # 载入再学习
    agent.train()
    train_off_policy(env, agent, cfg, done_add=False)
    agent.load_model(cfg.save_path)
    agent.eval()
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)


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


if __name__ == '__main__':
    # BipedalWalkerHardcore_TD3_test()
    # test_env()
    CarRacing_TD3_test()
    

