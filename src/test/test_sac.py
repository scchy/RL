
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


def sac_test_new():
    """
    policyNet: 
    valueNet: 
    """
    env_name = 'Pendulum-v1'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode = 220,
        save_path=os.path.join(path_, "test_models" ,f'SAC_{env_name}.ckpt'), 
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[64, 32],
        actor_lr=3e-5,
        critic_lr=5e-4,
        sample_size=256,
        off_buffer_size=10240,
        off_minimal_size=1024,
        max_episode_rewards=2048,
        max_episode_steps=240,
        gamma=0.95,
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
    # train_off_policy(env, agent, cfg, test_episode_count=5)
    try:
        agent.target_q.load_state_dict(
            torch.load(cfg.save_path)
        )
    except Exception as e:
        agent.actor.load_state_dict(
            torch.load(cfg.save_path)
        )
    env_ = gym.make(env_name, render_mode='human')
    play(env_, agent, cfg, episode_count=2, render=True)



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
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
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
        num_episode=2500,
        save_path=os.path.join(path_, "test_models" ,'SAC_Pusher-v4_test_actor-3.ckpt'), 
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
        alpha_lr=5e-3,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    # Episode [ 1867 / 3000|(seed=8526) ]:  62%|██████████▌      | 1866/3000 [4:35:15<3:16:03, 10.37s/it, steps=800, lastMeanRewards=-149.66, BEST=-119.49]
    # agent.actor.load_state_dict(
    #     torch.load(os.path.join(path_, "test_models" ,'SAC_Pusher-v4_test_actor-2.ckpt'), map_location='cpu')
    # )
    # train_off_policy(env, agent, cfg, train_without_seed=True, wandb_flag=True, 
    #                  step_lr_flag=True, step_lr_kwargs={'step_size': 1000, 'gamma': 0.9})
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    cfg.max_episode_steps = 200
    play(gym.make(env_name, render_mode='human'), agent, cfg, episode_count=2, play_without_seed=True)



def sac_Walker2d_v4_test():
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=2500,
        save_path=os.path.join(path_, "test_models" ,f'SAC_Pusher-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=4.5e-5,
        critic_lr=5.5e-4,
        sample_size=256,
        off_buffer_size=204800*4,
        off_minimal_size=2048 * 10,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        SAC_kwargs={
            'tau': 0.005, # soft update parameters
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
        alpha_lr=5e-3,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    # print(cfg.SAC_kwargs, env.action_space.shape)
    # train_off_policy(
    #     env, agent, cfg, train_without_seed=True, wandb_flag=False, 
    #     wandb_project_name=f'SAC-{env_name}',
    #     test_episode_count=5,
    #     step_lr_flag=True, 
    #     step_lr_kwargs={'step_size': 1000, 'gamma': 0.9}
    # )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    agent.eval()
    cfg.max_episode_steps = 200
    env = gym.make(env_name) #, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)


# swimmer
def sac_swimmer_test():
    env_name = 'Swimmer-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=1800,
        save_path=os.path.join(path_, "test_models" ,f'SAC-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=2.5e-5,
        critic_lr=5.5e-4,
        sample_size=256,
        off_buffer_size=204800,
        off_minimal_size=2048,
        max_episode_rewards=2048,
        max_episode_steps=60,
        gamma=0.98,
        SAC_kwargs={
            'tau': 0.005, # soft update parameters
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
        alpha_lr=5e-3,
        gamma=cfg.gamma,
        SAC_kwargs=cfg.SAC_kwargs,
        device=cfg.device
    )
    print(cfg.SAC_kwargs, env.action_space.shape)
    train_off_policy(
        env, agent, cfg, train_without_seed=True, wandb_flag=False, 
        wandb_project_name=f'SAC-{env_name}',
        test_episode_count=5,
        step_lr_flag=True, 
        step_lr_kwargs={'step_size': 1000, 'gamma': 0.9}
    )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    cfg.max_episode_steps = 200
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=True)


if __name__ == '__main__':
    # sac_test()
    # sac_Reacher_v4_test()
    # sac_Pusher_v4_test()
    sac_Walker2d_v4_test() # 2025-06-11
    # sac_swimmer_test()