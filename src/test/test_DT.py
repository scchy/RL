
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
from RLAlgo.batchRL.decision_tf import DTAgent
from RLUtils.batchRL.trainer import  DT_training, play, logger, dt_play
from RLUtils import Config, gym_env_desc
from RLUtils.env_wrapper import FrameStack, baseSkipFrame, GrayScaleObservation, ResizeObservation


def dt_Walker2d_v4_simple_test():
    data_level = 'simple' 
    add_str_ = '' if data_level == 'simple' else f'-{data_level}'
    env_name = 'Walker2d-v4'
    gym_env_desc(env_name)
    env = gym.make(env_name)
    print("gym.__version__ = ", gym.__version__ )
    path_ = os.path.dirname(__file__)
    cfg = Config(
        env, 
        save_path=os.path.join(path_, "test_models" ,f'DT-{env_name}{add_str_}'), 
        learning_rate=1.5e-4, 
        max_episode_rewards=2048,
        max_episode_steps=1000,
        max_ep_len=1000,
        num_epoches=1680,
        K=30,
        batch_size=20 * 32,
        target_return=5000,
        DT_kwargs=dict(
            embed_dim=128,
            n_layer=3,
            n_head=1,
            activation_function='relu',
            dropout=0.1,
            rtg_scale=1000,
            norm_obs=True
        )
    )
    agent = DTAgent(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim, 
        learning_rate=cfg.learning_rate,
        K=cfg.K,
        max_ep_len=cfg.max_episode_steps,
        DT_kwargs=cfg.DT_kwargs,
        device=cfg.device
    )
    DT_training(
        agent, 
        cfg,
        env_name,
        data_level=data_level, 
        test_episode_freq=10,
        episode_count=5,
        play_without_seed=True, 
        render=False
    )
    agent.load_model(cfg.save_path)
    logger.info('--'*25 + ' [ EVALUATION-PLAY ] ' + '--'*25)
    agent.eval()
    env = gym.make(env_name) #, render_mode='human')
    dt_play(env, agent, cfg, episode_count=2, play_without_seed=True, render=False)


if __name__ == '__main__':
    dt_Walker2d_v4_simple_test()

