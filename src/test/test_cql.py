
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
from RLUtils.batchRL.trainer import  batch_rl_training, play, logger
from RLUtils import Config, gym_env_desc
from RLUtils.env_wrapper import FrameStack, baseSkipFrame, GrayScaleObservation, ResizeObservation




def cql_Walker2d_v4_simple_test():
    data_level = 'simple' 
    add_str_ = '' if data_level == 'simple' else f'-{data_level}'
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}{add_str_}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=2.5e-4, # simple
        critic_lr=4.5e-4, # simple
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=3600, # simple
        batch_size=256,
        CQL_kwargs=dict(
            temp=1.2,
            min_q_weight=1.0, # simple
            num_random=10,
            tau=0.05, # simple
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=2.5, # simple
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3, #simple 
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device,
        reward_func=lambda r: (r + 10)/10 # simple
    )
    batch_rl_training(
        agent, 
        cfg,
        env_name,
        data_level=data_level, 
        test_episode_freq=10,
        episode_count=5,
        play_without_seed=True, 
        render=False
    )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    logger.info('--'*25 + ' [ EVALUATION-PLAY ] ' + '--'*25)
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name) #, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)


def cql_Walker2d_v4_medium_test():
    """ 
    When should we prefer Decision Transformers for Offline Reinforcement Learning?
        https://arxiv.org/abs/2305.14550
    
    medium-v0 数据集:
    数据来源: 使用中等水平训练的策略生成的数据。
    数据质量: 数据质量中等,策略性能大约为专家策略的 1/3。
    CQL 的劣势 :在中等质量数据上,CQL 的表现可能不如在高质量或低质量数据上稳定。
               这是因为中等质量数据中包含更多策略的混合, 导致 CQL 的保守性正则化难以有效工作。

    2. 策略多样性的影响
    单一策略 vs 混合策略:
    单一策略: 在单一策略(如随机策略或专家策略)生成的数据集中,CQL 的保守性正则化能够更好地发挥作用。
    混合策略: 在混合策略生成的数据集中,CQL 的保守性正则化可能无法有效区分不同策略的质量,导致性能下降。

    3. 实验结果
    实验表明:
        CQL 在低质量数据集(如 simple-v0)上表现较好,但在中等质量数据集(如 medium-v0)上表现不佳。
        这可能是因为中等质量数据集中的策略多样性增加了 CQL 的保守性正则化的难度。
    """
    # reference:  https://github.com/opendilab/DI-engine/blob/main/dizoo/d4rl/config/walker2d_medium_cql_config.py
    data_level = 'medium' 
    add_str_ = '' if data_level == 'simple' else f'-{data_level}'
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}{add_str_}.ckpt'), 
        actor_hidden_layers_dim=[256, 256],
        critic_hidden_layers_dim=[256, 256],
        actor_lr=1.5e-4, # 2.5e-4, # simple
        critic_lr=4.5e-4, # 4.5e-4, # simple
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=3600, 
        batch_size=256,
        CQL_kwargs=dict(
            temp=1.2,
            min_q_weight=12.75, # 混合策略 增加min_q 约束 12.75
            num_random=10,
            tau=0.05,  # 0.05
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=1.25 
            # min_q_weight=12.75 & reward_scale=1.25,   -> 577.34  
            # X min_q_weight=12.75 & reward_scale=1.125,  -> 564.74. 
            # X min_q_weight=12.75 & reward_scale=10.25   -> 390.63.
            # X min_q_weight=7.75 & reward_scale=1.25,   -> 433
            # X min_q_weight=17.75 & reward_scale=1.25,   -> 487 - 560.54.
            # min_q_weight=10.75 & reward_scale=1.25,   -> 192.82.
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3, #simple 
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device,
        # reward_func=lambda r: (r + 10)/10 # simple
    )
    batch_rl_training(
        agent, 
        cfg,
        env_name,
        data_level=data_level, 
        test_episode_freq=10,
        episode_count=5,
        play_without_seed=True, 
        wandb_flag=True,
        render=False,
        wandb_project_name=f'OfflineRL-{env_name}'
    )
    agent.actor.load_state_dict(
        torch.load(cfg.save_path, map_location='cpu')
    )
    logger.info('--'*25 + ' [ EVALUATION-PLAY ] ' + '--'*25)
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name) #, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)


def cql_Walker2d_v4_expert_bc_test():
    """
    BC 调参核心 norm_obs + lr 
    """
    data_level = 'expert' 
    add_str_ = '' if data_level == 'simple' else f'-{data_level}'
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}{add_str_}-BC'), 
        actor_hidden_layers_dim=[256, 256, 256],
        critic_hidden_layers_dim=[256, 256, 256],
        actor_lr=1.047e-4, # 1.05e-4 bc
        critic_lr=3.5e-4,  
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=6000,  # bc  6000
        batch_size=128,    # bc  128
        CQL_kwargs=dict(
            temp=1.0,
            min_q_weight=2.45,  
            num_random=10,
            tau=0.025,
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=1.25, 
            bc_flag=True,
            norm_obs=True
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3, #simple 
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device,
        #reward_func=lambda r: (r + 10)/10 
    )
    # batch_rl_training(
    #     agent, 
    #     cfg,
    #     env_name,
    #     data_level=data_level, 
    #     test_episode_freq=10,
    #     episode_count=5,
    #     play_without_seed=True, 
    #     wandb_flag=True,
    #     render=False,
    #     wandb_project_name=f'OfflineRL-{env_name}-{data_level}',
    #     name_str='-bc' if cfg.CQL_kwargs.get('bc_flag', False) else ''
    # )
    agent.load_model(cfg.save_path)
    logger.info('--'*25 + ' [ EVALUATION-PLAY ] ' + '--'*25)
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=3, play_without_seed=True, render=True)


def cql_Walker2d_v4_expert_test():
    """
    BC 调参核心 norm_obs + lr 
    """
    data_level = 'expert' 
    add_str_ = '' if data_level == 'simple' else f'-{data_level}'
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'CQL-{env_name}{add_str_}-obsNorm'), 
        actor_hidden_layers_dim=[256, 256, 256],
        critic_hidden_layers_dim=[256, 256, 256],
        actor_lr=2.25e-4,  # 1.047e-4, 
        critic_lr=4.5e-4,  
        max_episode_rewards=2048,
        max_episode_steps=800,
        gamma=0.98,
        num_epoches=21600,   # 12600,  
        batch_size=128,    
        CQL_kwargs=dict(
            temp=1.0,
            min_q_weight=2.45,  
            num_random=10,
            tau=0.025,
            target_entropy=-torch.prod(torch.Tensor(env.action_space.shape)).item(),
            action_bound=1.0,
            reward_scale=1.25, 
            bc_flag=False,
            norm_obs=True
        )
    )
    agent = CQL(
        state_dim=cfg.state_dim,
        actor_hidden_layers_dim=cfg.actor_hidden_layers_dim,
        critic_hidden_layers_dim=cfg.critic_hidden_layers_dim,
        action_dim=cfg.action_dim, 
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=5e-3, #simple 
        gamma=cfg.gamma,
        CQL_kwargs=cfg.CQL_kwargs,
        device=cfg.device,
        #reward_func=lambda r: (r + 10)/10 
    )
    # batch_rl_training(
    #     agent, 
    #     cfg,
    #     env_name,
    #     data_level=data_level, 
    #     test_episode_freq=10,
    #     episode_count=5,
    #     play_without_seed=True, 
    #     wandb_flag=True,
    #     render=False,
    #     wandb_project_name=f'OfflineRL-{env_name}-{data_level}',
    #     name_str='-bc' if cfg.CQL_kwargs.get('bc_flag', False) else ''
    # )
    agent.load_model(cfg.save_path)
    logger.info('--'*25 + ' [ EVALUATION-PLAY ] ' + '--'*25)
    agent.eval()
    cfg.max_episode_steps = 600
    env = gym.make(env_name, render_mode='human')
    play(env, agent, cfg, episode_count=2, play_without_seed=True, render=True)


if __name__ == '__main__':
    # cql_Walker2d_v4_simple_test()
    # cql_Walker2d_v4_medium_test()
    # cql_Walker2d_v4_expert_bc_test()
    cql_Walker2d_v4_expert_test()

