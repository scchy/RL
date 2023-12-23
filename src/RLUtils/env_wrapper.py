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













