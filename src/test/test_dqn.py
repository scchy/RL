
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
    env_name = 'Pendulum-v1' # num_episode=100 action_contiguous_ = True
    env_name = 'CartPole-v1' # num_episode=300 action_contiguous_ = False
    gym_env_desc(env_name)
    env = gym.make(env_name)
    action_contiguous_ = False
    cfg = Config(
        env, 
        num_episode=300,
        save_path=r'D:\TMP\dqn_target_q.ckpt',
        hidden_layers_dim=[10, 10],
        split_action_flag=True,
        target_update_freq=3,
        sample_size=256,
        off_buffer_size=20480,
        max_episode_rewards=260,
        max_episode_steps=260,
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
    train_off_policy(env, dqn, cfg, action_contiguous=action_contiguous_)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2, action_contiguous=action_contiguous_)


if __name__ == '__main__':
    dqn_test()

