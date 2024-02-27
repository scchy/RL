# python3
# Create date: 2023-12-02
# Author: Scc_hy
# Func: pic-observe preprocessing
# =================================================================================

import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack, LazyFrames
from collections import deque
from typing import List, Dict


class baseSkipFrame(gym.Wrapper):
    def __init__(
            self, 
            env, 
            skip: int, 
            cut_slices: List[List[int]]=None,
            start_skip: int=None,
            neg_action_kwargs: Dict=None
        ):
        """_summary_

        Args:
            env (_type_): _description_
            skip (int): skip frames
            cut_slices (List[List[int]], optional): pic observation cut. Defaults to None.
            start_skip (int, optional): skip several frames to start. Defaults to None.
            neg_action_kwargs (Dict): {action: neg_reward} 对某些动作予以负分
        """
        super().__init__(env)
        self._skip = skip
        self.pic_cut_slices = cut_slices
        self.start_skip = start_skip
        self.neg_action_kwargs = neg_action_kwargs
    
    def _cut_slice(self, obs):
        if self.pic_cut_slices is None:
            return obs
        slice_list = []
        for idx, dim_i_slice in enumerate(self.pic_cut_slices):
            slice_list.append(eval('np.s_[{st}:{ed}]'.format(st=dim_i_slice[0], ed=dim_i_slice[1])))
    
        obs = obs[tuple(i for i in slice_list)]
        return obs

    def step(self, action):
        neg_r = 0.0
        if self.neg_action_kwargs is not None:
            try:
                a_ = action[0]
            except Exception as e:
                a_ = int(action)
            neg_r = self.neg_action_kwargs.get(a_, 0.0)
        tt_reward_list = []
        done = False
        total_reward = 0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward += neg_r
            done_f = terminated or truncated
            total_reward += reward
            tt_reward_list.append(reward)
            if done:
                break
        
        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        return obs, total_reward, done_f, truncated, info
    
    def _start_skip(self):
        a = np.array([0.0, 0.0, 0.0]) if hasattr(self.env.action_space, 'low') else np.array(0) 
        for i in range(self.start_skip):
            obs, reward, terminated, truncated, info = self.env.step(a)
        return obs, info

    def reset(self, seed=0, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        if self.start_skip is not None:
            obs, info = self._start_skip()
        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        return obs, info


class CarV2SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
        """
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        tt_reward_list = []
        done = False
        total_reward = 0
        for i in range(self._skip):
            obs, reward, done, info, _ = self.env.step(action)
            out_done = self.judge_out_of_route(obs)
            done_f = done or out_done
            reward = -10 if out_done else reward
            # reward = -100 if out_done else reward
            # reward = reward * 10 if reward > 0 else reward
            total_reward += reward
            tt_reward_list.append(reward)
            if done_f:
                break
        return obs[:84, 6:90, :], total_reward, done_f, info, _
    
    def judge_out_of_route(self, obs):
        s = obs[:84, 6:90, :]
        out_sum = (s[75, 35:48, 1][:2] > 200).sum() + (s[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4

    def reset(self, seed=0, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        # steering  gas  breaking
        a = np.array([0.0, 0.0, 0.0])
        for i in range(45):
            obs, reward, done, info, _ = self.env.step(a)

        return obs[:84, 6:90, :], info


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
        """
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info, _


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """RGP -> Gray
        (high, width, channel) -> (1, high, width) 
        """
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8
        )
    
    def observation(self, observation):
        tf = transforms.Grayscale()
        # channel first
        return tf(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape: int):
        """reshape observe
        Args:
            env (_type_): _description_
            shape (int): reshape size
        """
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        #  Normalize -> input[channel] - mean[channel]) / std[channel]
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)













