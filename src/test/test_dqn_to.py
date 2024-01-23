
import os
from os.path import dirname
import sys
import gym
import torch

try:
    dir_ = dirname(dirname(__file__))
except Exception as e:
    dir_ = dirname(dirname('__file__'))

if len(dir_) == 0:
    dir_ = os.getcwd() + '/src'
print(dir_)
sys.path.append(dir_)
from RLAlgo.DQN import DQN
from RLUtils import train_off_policy, play, Config, gym_env_desc
from RLUtils.env_wrapper import FrameStack, baseSkipFrame, GrayScaleObservation, ResizeObservation



def DemonAttack_v5_dqn_new_test():
    env_name = 'ALE/DemonAttack-v5' 
    gym_env_desc(env_name)
    env = gym.make(env_name, obs_type="rgb")
    print("gym.__version__ = ", gym.__version__ )
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=5, 
                cut_slices=[[15, 188], [0, 160]],
                start_skip=14)), 
            shape=84
        ), 
        num_stack=4
    )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        # 环境参数
        split_action_flag=False,
        save_path=os.path.join(path_, "test_models" ,f'dqn_DemonAttack-v5-new_1'),
        seed=42,
        # 网络参数
        hidden_layers_dim=[200, 200],
        # agent参数
        learning_rate=1.0e-4,
        target_update_freq=16,
        gamma=0.99,
        epsilon=0.05,
        # 训练参数
        num_episode=1500,
        off_buffer_size=12000,
        off_minimal_size=1024, 
        sample_size=32,
        max_episode_steps=280,
        # agent 其他参数
        dqn_type = 'DoubleDQN-CNN',
        epsilon_start=0.95,
        epsilon_decay_steps=15000
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
        dqn_type=cfg.dqn_type,
        epsilon_start=cfg.epsilon_start,
        epsilon_decay_steps=cfg.epsilon_decay_steps
    )
    dqn.load_model(cfg.save_path)
    dqn.eval()
    env = gym.make(env_name, obs_type="rgb")# render_mode='human')
    # env = gym.make(env_name, obs_type="rgb", render_mode='human')
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=5, 
                cut_slices=[[15, 188], [0, 160]],
                start_skip=14)), 
            shape=84
        ), 
        num_stack=4
    )
    play(env, dqn, cfg, episode_count=3, 
         play_without_seed=True, render=False)
    # play(env, dqn, cfg, episode_count=1, 
    #      play_without_seed=True, render=False)



if __name__ == '__main__':
    # LunarLander_dqn_test()
    # DemonAttack_v5_dqn_test()
    DemonAttack_v5_dqn_new_test()

