
import os
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.DQN import DQN
from RLUtils import train_off_policy, play, Config, gym_env_desc


def dqn_test():
    # num_episode=100 action_contiguous_ = True
    env_name = 'Pendulum-v1' 
    # num_episode=500 action_contiguous_ = False epsilon=0.01
    env_name = 'CartPole-v1' 
    # 要需要提升探索率，steps需要足够大
    # num_episode=200 action_contiguous_ = False epsilon=0.05 max_episode_steps=500
    env_name = 'MountainCar-v0' 
    gym_env_desc(env_name)
    env = gym.make(env_name)
    action_contiguous_ = False # 是否将连续动作离散化

    cfg = Config(
        env, 
        # 环境参数
        split_action_flag=True,
        save_path=r'D:\TMP\dqn_target_q.ckpt',
        seed=42,
        # 网络参数
        hidden_layers_dim=[32, 32],
        # agent参数
        learning_rate=2e-3,
        target_update_freq=3,
        gamma=0.95,
        epsilon=0.05,
        # 训练参数
        num_episode=200,
        off_buffer_size=2048+1024,
        off_minimal_size=1024,
        sample_size=256,
        max_episode_steps=500,
        # agent 其他参数
        dqn_type = 'duelingDQN'
    )
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    # train_off_policy(env, dqn, cfg, action_contiguous=action_contiguous_)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2, action_contiguous=action_contiguous_)




def Acrobot_dqn_test():
    # num_episode=100 action_contiguous_ = True
    env_name = 'Acrobot-v1' 
    gym_env_desc(env_name)
    env = gym.make(env_name)
    action_contiguous_ = False # 是否将连续动作离散化

    cfg = Config(
        env, 
        # 环境参数
        split_action_flag=True,
        save_path=r'D:\TMP\Acrobot_dqn_target_q.ckpt',
        seed=42,
        # 网络参数
        hidden_layers_dim=[128, 64],
        # agent参数
        learning_rate=2e-3,
        target_update_freq=3,
        gamma=0.95,
        epsilon=0.05,
        # 训练参数
        num_episode=300,
        off_buffer_size=20480,
        off_minimal_size=1024,
        sample_size=256,
        max_episode_steps=400,
        # agent 其他参数
        dqn_type = 'duelingDQN'
    )
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    # train_off_policy(env, dqn, cfg, action_contiguous=action_contiguous_)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2, action_contiguous=action_contiguous_)




def LunarLander_dqn_test():
    # num_episode=100 action_contiguous_ = True
    env_name = 'LunarLander-v2' 
    #  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pybox2d 下载包
    # pip install + 路径/xxx.whl
    gym_env_desc(env_name)
    env = gym.make(env_name)
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        split_action_flag=True,
        save_path=os.path.join(path_, "test_models" ,f'dqn_{env_name}_1.ckpt'), 
        seed=42,
        # 网络参数
        hidden_layers_dim=[128, 64],
        # agent参数
        learning_rate=2e-3,
        target_update_freq=3,
        gamma=0.99,
        epsilon=0.05,
        # 训练参数
        num_episode=800,
        off_buffer_size=20480,
        off_minimal_size=2048,
        sample_size=128,
        max_episode_steps=200,
        # agent 其他参数
        dqn_type = 'duelingDQN'
    )
    dqn = DQN(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        target_update_freq=cfg.target_update_freq,
        device=cfg.device,
        dqn_type=cfg.dqn_type
    )
    # train_off_policy(env, dqn, cfg)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2)


if __name__ == '__main__':
    LunarLander_dqn_test()

