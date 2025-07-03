# python3
# Create Date: 2025-07-03
# Author: Scc_hy
# Func: Batch RL utils
# ===================================================================================

import os 
import time
import sys 
import re
import numpy as np 
import gymnasium as gym
import minari
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
# export D4RL_DATASET_DIR=/home/scc/sccWork/devData/sccDisk/offline_rl_data
data_dir = os.environ.get('MINARI_DATASETS_PATH')
# env_name = 'Walker2d-v4'
# env = gym.make(env_name)
# data_name = 'dataset_farama-minari_mujoco/walker2d/simple-v0'
# dataset = minari.load_dataset(data_name)
# dir(dataset)


def mujoco_download():
    data_name = 'farama-minari/mujoco'
    username, repository = data_name.split('/')
    final_out_path = p_join(data_dir, f'dataset_{username}_{repository}')
    os.system(f"""
    export HF_ENDPOINT=https://hf-mirror.com && \
    huggingface-cli download --resume-download {data_name} \
    --local-dir-use-symlinks False \
    --repo-type dataset \
    --local-dir {final_out_path} \
    --cache-dir {final_out_path}/cache
    """)
    os.system(f'rm -rf {final_out_path}/cache')


def load_mujoco_data(env_name, level='simple'):
    env_str = env_name.split('/')[-1].split('-')[0].lower()
    data_name = f'dataset_farama-minari_mujoco/{env_str}/{level}-v0'
    return minari.load_dataset(data_name)


class epDataset(Dataset):
    def __init__(self, episode_data):
        super(epDataset, self).__init__()
        state_tt = episode_data.observations
        self.state = state_tt[:-1]
        self.action = episode_data.actions
        self.reward = episode_data.rewards
        self.next_state = state_tt[1:]
        self.done = np.logical_or(episode_data.terminations, episode_data.truncations) * 1
    
    def __len__(self):
        return len(self.action)

    def __getitem__(self, index):
        states = self.state[index]
        actions = self.action[index]
        reward = self.reward[index]
        next_state = self.next_state[index]
        done = self.done[index]
        return states, actions, reward, next_state, done


if __name__ == '__main__':
    env_name = 'Walker2d-v4'
    data_ = load_mujoco_data(env_name)
    for episode_data in data_.iterate_episodes():
        dloader = DataLoader(epDataset(episode_data), batch_size=256, shuffle=True)
        for batch in dloader:
            states, actions, reward, next_state, done = batch
            print(f'{states.shape=} {actions.shape=} {reward.shape=}')
        break 


