
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



def DemonAttack_v5_dqn_test():
    env_name = 'ALE/DemonAttack-v5' 
    gym_env_desc(env_name)
    env = gym.make(env_name, obs_type="rgb")
    print("gym.__version__ = ", gym.__version__ )
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=4, 
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
        save_path=os.path.join(path_, "test_models" ,f'dqn_DemonAttack-v5_1'),
        seed=42,
        # 网络参数
        hidden_layers_dim=[200, 200],
        # agent参数
        learning_rate=3.0e-4,
        target_update_freq=8,
        gamma=0.99,
        epsilon=0.05, # 0.1
        # 训练参数
        num_episode=2000,
        off_buffer_size=12000,
        off_minimal_size=2048, 
        sample_size=32,
        max_episode_steps=260,
        # agent 其他参数
        dqn_type = 'DoubleDQN-CNN'
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
    # dqn.train()
    # train_off_policy(env, dqn, cfg, done_add=False, train_without_seed=False, wandb_flag=False, test_ep_freq=50)
    dqn.load_model(cfg.save_path)
    dqn.eval()
    env = gym.make(env_name, obs_type="rgb") #, render_mode='human')
    env = FrameStack(
        ResizeObservation(
            GrayScaleObservation(baseSkipFrame(
                env, 
                skip=4, 
                cut_slices=[[15, 188], [0, 160]],
                start_skip=14)), 
            shape=84
        ), 
        num_stack=4
    )
    play(env, dqn, cfg, episode_count=10, render=False)
    record = """
    1. [ X ] ADD max_episode_steps 200 -> 500 : not work well
    2. [ X ] ADD max_episode_steps 200 -> 500 
        + ADD hidden_layers_dim size [200, 200] -> [256, 256] 
        + REDUCE sample_size 128 -> 64 : not work well
    3. [ √ ]  REDUCE learning_rate 2.5e-4 -> 3.0e-4 
        +   ADD max_episode_steps 200 -> 260 
        +   REDUCE sample_size 128 ->  32
    """


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
        learning_rate=2.5e-4,
        target_update_freq=16,
        gamma=0.99,
        epsilon=0.05,
        # 训练参数
        num_episode=1200,
        off_buffer_size=12000,
        off_minimal_size=1024, 
        sample_size=32,
        max_episode_steps=260,
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
    dqn.train()
    train_off_policy(env, dqn, cfg, done_add=False, 
                     train_without_seed=True, 
                     wandb_flag=False, 
                     test_ep_freq=50, test_episode_count=10)
    dqn.load_model(cfg.save_path)
    dqn.eval()
    env = gym.make(env_name, obs_type="rgb")#, render_mode='human')
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
    play(env, dqn, cfg, episode_count=1, 
         play_without_seed=True, render=False)


if __name__ == '__main__':
    # LunarLander_dqn_test()
    # DemonAttack_v5_dqn_test()
    DemonAttack_v5_dqn_new_test()

