
from os.path import dirname
import sys
import gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.PPO import PPO
from RLUtils import train_on_policy, play, Config, gym_env_desc


def r_func(r):
    # 不希望留在谷底
    return (r + 10.0 ) / 10.0


def ppo_test():
    """
    policyNet: 
    valueNet: 
    """
    # A:[128, 64] C:[64, 32]
    env_name = 'Pendulum-v1'
    # env_name = 'MountainCarContinuous-v0'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    cfg = Config(
        env, 
        # 环境参数
        save_path=r'D:\TMP\ppo_test_actor.ckpt', 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[64, 32],
        # agent参数
        actor_lr=1e-4,
        critic_lr=5e-3,
        gamma=0.9,
        # 训练参数
        num_episode=1200,
        off_buffer_size=204800,
        max_episode_steps=260,
        # agent其他参数
        PPO_kwargs={
            'lmbda': 0.9,
            'eps': 0.2, # clip eps
            'k_epochs': 10,
            'sgd_batch_size': 512,
            'minibatch_size': 128,
            'actor_nums': 3,
            'actor_bound': 2 # tanh (-1, 1)
        }
    )
    agent = PPO(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        reward_func=r_func
    )
    train_on_policy(env, agent, cfg)
    agent.actor.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2)


if __name__ == '__main__':
    ppo_test()

