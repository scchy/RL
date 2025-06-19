
import os
from os.path import dirname
import sys
import gymnasium as gym
import torch
dir_ = dirname(dirname(__file__))
print(dir_)
sys.path.append(dir_)
from RLAlgo.SoftQ import SQL
from RLAlgo.SoftQNew import SoftQ as SQLNew
from RLUtils import train_off_policy, play, Config, gym_env_desc


def sql_test():
    env_name = 'Pendulum-v1'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=220,
        save_path=os.path.join(path_, "test_models" ,f'SQL-{env_name}.ckpt'), 
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
        SQL_kwargs={
            'tau': 0.05,
            'action_bound': 2.0,
            'alpha': 0.25
        }
    )
    agent = SQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        SQL_kwargs=cfg.SQL_kwargs,
        device=cfg.device
    )
    # train_off_policy(
    #     env, agent, cfg, train_without_seed=True, wandb_flag=False, 
    #     wandb_project_name=f'SQL-{env_name}',
    #     test_episode_count=5
    # )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    cfg.max_episode_steps = 200
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=True)



def sqlnew_test():
    env_name = 'Pendulum-v1'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=500,
        save_path=os.path.join(path_, "test_models" ,f'SQL-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[128, 64],
        critic_hidden_layers_dim=[128, 64],
        actor_lr=1e-4,
        critic_lr=3e-4,
        sample_size=256,
        off_buffer_size=10240,
        off_minimal_size=1024,
        max_episode_rewards=2048,
        max_episode_steps=240,
        gamma=0.95,
        SQL_kwargs=dict(
            value_n_particles=16,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            critcic_traget_update_freq=100
        )
    )
    agent = SQLNew(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim, 
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        SQL_kwargs=cfg.SQL_kwargs,
        device=cfg.device
    )
    # train_off_policy(
    #     env, agent, cfg, train_without_seed=True, wandb_flag=False, 
    #     wandb_project_name=f'SQL-{env_name}',
    #     test_episode_count=5
    # )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    agent.eval()
    cfg.max_episode_steps = 200
    env = gym.make(env_name) #, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)
 
 
def sqlnew_Walker2d_v4_test():
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        num_episode=1500,
        save_path=os.path.join(path_, "test_models" ,f'SQL-{env_name}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=1.5e-4,
        critic_lr=5.5e-4,
        sample_size=256,
        off_buffer_size=204800*2,
        off_minimal_size=2048*2,
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.99,
        SQL_kwargs=dict(
            value_n_particles=16,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            critcic_traget_update_freq=100,
            reward_scale=1
        )
    )
    agent = SQLNew(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim, 
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        SQL_kwargs=cfg.SQL_kwargs,
        device=cfg.device
    )
    # train_off_policy(
    #     env, agent, cfg, train_without_seed=True, wandb_flag=False, 
    #     wandb_project_name=f'SQL-{env_name}',
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



if __name__ == '__main__':
    # sql_test()
    # sqlnew_test() # 2025-06-18
    sqlnew_Walker2d_v4_test()

