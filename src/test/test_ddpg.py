
import os
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.DDPG import DDPG
from RLUtils import train_off_policy, play, Config




def ddpg_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    cfg = Config(
        env, 
        num_episode = 180,
        save_path=r'D:\TMP\ddpg_test_actor.ckpt', 
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[64, 32],
        actor_lr=3e-5,
        critic_lr=5e-4,
        sample_size=256,
        off_buffer_size=2048,
        off_minimal_size=1024,
        max_episode_rewards=2048,
        max_episode_steps=240,
        gamma=0.9,
        DDPG_kwargs={
            'tau': 0.05, # soft update parameters
            'sigma': 0.005, # noise
            'action_bound': 2.0
        }
    )
    agent = DDPG(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        DDPG_kwargs=cfg.DDPG_kwargs,
        device=cfg.device
    )
    train_off_policy(env, agent, cfg)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)


if __name__ == '__main__':
    ddpg_test()

