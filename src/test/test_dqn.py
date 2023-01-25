
import os
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.DQN import DQN
from RLUtils import train_off_policy, play, Config

    

def dqn_test():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    cfg = Config(
        env, 
        split_action_flag=True,
        target_update_freq=3,
        save_path=r'D:\TMP\dqn_target_q.ckpt',
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
    train_off_policy(env, dqn, cfg, action_contiguous=True)
    dqn.target_q.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), dqn, cfg, episode_count=2, action_contiguous=True)


if __name__ == '__main__':
    dqn_test()

