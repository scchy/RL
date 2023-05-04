
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



def reward_func(r, d):
    if r <= -100:
        r = -1
        d = True
    else:
        d = False
    return r, d


def LunarLanderContinuous_ddpg_test():
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
        save_path=os.path.join(path_, "test_models" ,'TD3_BipedalWalkerHardcore-v3_test_actor.ckpt'), 
        seed=42,
        # 网络参数
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        # agent参数
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        # 训练参数
        num_episode=5000,
        sample_size=256,
        off_buffer_size=int(1e6),
        off_minimal_size=2048,
        max_episode_rewards=500,
        max_episode_steps=300,
        # agent 其他参数
        TD3_kwargs={
            'tau': 0.005, # soft update parameters
            'sigma': 0.5, # noise
            'action_bound': 1.0,
            'action_low': env.action_space.low[0],
            'action_high': env.action_space.high[0],
            'expl_noise': 0.25
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


if __name__ == '__main__':
    LunarLanderContinuous_ddpg_test()

