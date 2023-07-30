
import os
from os.path import dirname
import sys
import gymnasium as gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.SAC import SAC
from RLUtils import train_off_policy, play, Config, gym_env_desc



def sac_test():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    cfg = Config(
        env, 
        num_episode = 100,
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
        SAC_kwargs={
            'tau': 0.05, # soft update parameters
            'target_entropy': 0.01,
            'action_bound': 2.0
        }
    )
    agent = SAC(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-4,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
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


def sac_Reacher_v4_test():
    """
    policyNet: 
    valueNet: 
    install https://blog.csdn.net/Scc_hy/article/details/131819065?spm=1001.2014.3001.5502
    mujoco210
        # 1- download  mujoco210
        cd /home/scc/ && makdir  .mujoco && cd  .mujoco
        wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
        tar xvf mujoco210-linux-x86_64.tar.gz
        # 2- set env PATH
        vi ~/.bashrc
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/scc/.mujoco/mujoco210/bin
            export PATH=/usr/local/bin:$PATH
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
        # 3- pip install
        pip install -U 'mujoco-py<2.2,>=2.1'
    
    final fix:
        pip install gymnasium
        pip install gymnasium[mujoco]
    """
    env_name = 'Reacher-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=5000,
        save_path=os.path.join(path_, "test_models" ,'SAC_Reacher-v4_test_actor-0.ckpt'), 
        actor_hidden_layers_dim=[200, 200],
        critic_hidden_layers_dim=[200, 200],
        actor_lr=1e-5,
        critic_lr=3e-4,
        sample_size=256,
        off_buffer_size=204800,
        off_minimal_size=2048,
        max_episode_rewards=2048,
        max_episode_steps=300,
        gamma=0.98,
        SAC_kwargs={
            'tau': 0.03, # soft update parameters
            'target_entropy': -torch.prod(torch.Tensor(env.action_space.shape)).item(),
            'action_bound': 1.0
        }
    )
    agent = SAC(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-4,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    # train_off_policy(env, agent, cfg, train_without_seed=True)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2, play_without_seed=True)


def sac_Pusher_v4_test():
    env_name = 'Pusher-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=3000,
        save_path=os.path.join(path_, "test_models" ,'SAC_Pusher-v4_test_actor-1.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=3e-5,
        critic_lr=5e-4,
        sample_size=256,
        off_buffer_size=204800*4,
        off_minimal_size=2048 * 10,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        SAC_kwargs={
            'tau': 0.005, # soft update parameters
            'target_entropy': -torch.prod(torch.Tensor(env.action_space.shape)).item(),
            'action_bound': 2.0
        }
    )
    agent = SAC(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-4,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    # Episode [ 390 / 3000|(seed=6282) ]:  13%|████▏                           | 389/3000 [34:20<5:00:19,  6.90s/it, steps=800, lastMeanRewards=-262.66, BEST=-251.57]
    train_off_policy(env, agent, cfg, train_without_seed=True)
    agent.actor.load_state_dict(
        torch.load(cfg.save_path)
    )
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2, play_without_seed=True)



if __name__ == '__main__':
    # sac_test()
    # sac_Reacher_v4_test()
    sac_Pusher_v4_test()