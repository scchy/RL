
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
from RLAlgo.batchRL.cql import CQL_H_SAC as CQL
from RLUtils.batchRL.trainer import  batch_rl_training, play
from RLUtils import Config, gym_env_desc
from RLUtils.env_wrapper import FrameStack, baseSkipFrame, GrayScaleObservation, ResizeObservation




def cql_Walker2d_v4_test():
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=2.5e-4,
        critic_lr=4.5e-4,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=1200,
        batch_size=256,
        CQL_kwargs=dict(
            temp=1.2,
            min_q_weight=1.0,
            num_random=10,
            tau=0.05,
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=2.5
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3,
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device
    )
    
    # batch_rl_training(
    #     agent, 
    #     cfg,
    #     env_name,
    #     data_level='simple',# 'medium', #
    #     test_episode_freq=10,
    #     episode_count=5,
    #     play_without_seed=True, 
    #     render=False
    # )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=True)


if __name__ == '__main__':
    cql_Walker2d_v4_test()

