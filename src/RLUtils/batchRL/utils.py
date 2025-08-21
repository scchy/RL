# python3
# Create Date: 2025-07-03
# Author: Scc_hy
# Func: Batch RL utils
# ===================================================================================

import os 
from os.path import join as p_join
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


class rtgEpDataset(Dataset):
    def __init__(self, episode_data, K):
        super(rtgEpDataset, self).__init__()
        state_tt = episode_data.observations
        self.max_length = K
        self.state = state_tt[:-1]
        self.state_dim = self.state.shape[-1]
        self.action = episode_data.actions
        self.action_dim = self.action.shape[-1]
        self.reward = episode_data.rewards
        self.next_state = state_tt[1:]
        self.done = np.logical_or(episode_data.terminations, episode_data.truncations) * 1
        self.ts = np.arange(len(self.action))
        self.rtg = np.cumsum(episode_data.rewards[::-1])[::-1]
        self.mask = np.ones(self.done.shape[0])
        # print(f'rtgEpDataset({self.rtg.max()=}, traj_len={self.state.shape[0]})')
        self._prepare_K()
    
    def _prepare_K(self):
        n_K_sample = self.state.shape[0] // self.max_length
        reshape_samples = n_K_sample * self.max_length
        state_ = self.state[:reshape_samples, ...].reshape(n_K_sample, self.max_length, self.state_dim)
        next_state_ = self.next_state[:reshape_samples, ...].reshape(n_K_sample, self.max_length, self.state_dim)

        action_ = self.action[:reshape_samples, ...].reshape(n_K_sample, self.max_length, self.action_dim)
        reward_ = self.reward[:reshape_samples].reshape(n_K_sample, self.max_length, 1)
        rtg_ = self.rtg[:reshape_samples].reshape(n_K_sample, self.max_length, 1)
        done_ = self.done[:reshape_samples].reshape(n_K_sample, self.max_length)
        ts_ = self.ts[:reshape_samples].reshape(n_K_sample, self.max_length)
        mask_ = np.ones((n_K_sample, self.max_length))

        self.state = np.concatenate([state_, self.padding_batch(self.state[np.newaxis, reshape_samples:, ...], pad_num=0.0, final_dim=self.state_dim)])
        self.next_state = np.concatenate([next_state_, self.padding_batch(self.next_state[np.newaxis, reshape_samples:, ...], pad_num=0.0, final_dim=self.state_dim)])
        self.action = np.concatenate([action_, self.padding_batch(self.action[np.newaxis, reshape_samples:, ...], pad_num=-10.0, final_dim=self.action_dim)])
        self.reward = np.concatenate([reward_, self.padding_batch(self.reward[np.newaxis, reshape_samples:, np.newaxis], pad_num=0.0, final_dim=1)])
        self.rtg = np.concatenate([rtg_, self.padding_batch(self.rtg[np.newaxis, reshape_samples:, np.newaxis], pad_num=0.0, final_dim=1)])
        self.done = np.concatenate([done_, self.padding_batch(self.done[np.newaxis, reshape_samples:], pad_num=2.0)])
        self.ts = np.concatenate([ts_, self.padding_batch(self.ts[np.newaxis, reshape_samples:], pad_num=0.0)])
        self.mask = np.concatenate([mask_, self.padding_batch(self.mask[np.newaxis, reshape_samples:], pad_num=0.0)])

    def padding_batch(self, t, pad_num=1.0, final_dim=None):
        if len(t) == 0:
            return np.array([])
        pad_nt = self.max_length - t.shape[1]
        if pad_nt == 0:
            return t 
        pad_np = np.ones((1, pad_nt)) * pad_num
        if final_dim is not None:
            pad_np = np.ones((1, pad_nt, final_dim)) * pad_num

        return np.concatenate([pad_np, t], axis=1)

    def __len__(self):
        return len(self.action)

    def __getitem__(self, index):
        states = self.state[index]
        actions = self.action[index]
        reward = self.reward[index]
        next_state = self.next_state[index]
        done = self.done[index]
        rtg = self.rtg[index]
        timesteps = self.ts[index]
        mask = self.mask[index]
        # s, a, r, ns, d, rtg, timesteps, mask
        return states, actions, reward, next_state, done, rtg, timesteps, mask
    

if __name__ == '__main__':
    env_name = 'Walker2d-v4'
    env = gym.make(env_name)
    print(f"{env.observation_space.shape=}")
    data_ = load_mujoco_data(env_name)
    idx = 0
    for episode_data in data_.iterate_episodes():
        if idx == 2:
            break
        print(f"len={len(episode_data.rewards)} {episode_data.rewards.sum()=}")
        dloader = DataLoader(epDataset(episode_data), batch_size=256, shuffle=True)
        for batch in dloader:
            states, actions, reward, next_state, done = batch
            print(f'[{idx=}] {states.shape=} {actions.shape=} {reward.shape=}')
        idx += 1

    data_ = load_mujoco_data(env_name)
    idx = 0
    for episode_data in data_.iterate_episodes():
        if idx == 2:
            break
        print(f"len={len(episode_data.rewards)} {episode_data.rewards.sum()=}")
        ds = rtgEpDataset(episode_data)
        dloader = DataLoader(ds, batch_size=256, shuffle=True)
        for batch in dloader:
            states, actions, reward, next_state, done, rtg, timesteps, mask = batch
            print(f'RTG [{idx=}] {states.shape=} {actions.shape=} {reward.shape=}')
        idx += 1

    print(ds.done, ds.reward, ds.rtg)
