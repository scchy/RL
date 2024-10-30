# python3
# Create date: 2023-12-02
# Author: Scc_hy
# Func: pic-observe preprocessing
# =================================================================================

import gymnasium as gym
import torch
import numpy as np
import cv2
import time 
from torchvision import transforms
from gymnasium.spaces import Box, Space
from gymnasium.wrappers import FrameStack, LazyFrames, RecordEpisodeStatistics
from collections import deque
from typing import List, Dict, Optional, Callable, Iterable, Tuple, Any, Sequence, Union
from numpy.typing import NDArray
from gymnasium import Env
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv
from copy import deepcopy


class spSyncVectorEnv(gym.vector.SyncVectorEnv):
    """
    step_await _terminateds reset
    """
    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
        random_reset: bool = False,
        seed: int = None
    ):
        super().__init__(env_fns, observation_space, action_space, copy)
        self.random_reset = random_reset
        self.seed = seed
    
    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i]:
                old_observation, old_info = observation, info
                if self.random_reset:
                    observation, info = env.reset(seed=np.random.randint(0, 999999))
                else:
                    observation, info = env.reset() if self.seed is None else env.reset(seed=self.seed) 
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )


class envPoolRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, envs, deque_size=100, max_no_reward_count: Optional[int]=None):
        super().__init__(envs)
        self.envs = envs
        self.num_envs = getattr(envs, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.record_episode_returns = None
        self.lives = None
        self.max_no_reward_count = max_no_reward_count
        self.max_no_reward_arr = None
        if max_no_reward_count is not None:
            self.max_no_reward_arr = np.ones(self.num_envs) * self.max_no_reward_count
        self.no_reward_count = np.zeros(self.num_envs)
        self.env_id_arr = np.arange(self.num_envs)

    def _one_env_reset(self, rewards, terminations, observations, infos, truncated):
        self.no_reward_count += (rewards == 0) * 1
        self.no_reward_count *= (rewards == 0) * 1 # 不等于0的重置
        zero_bool = infos['lives'] == 0
        truncated[zero_bool & terminations] = True
        if self.max_no_reward_count is not None:
            reset_bool = self.no_reward_count >= self.max_no_reward_arr
            if reset_bool.sum():
                self.no_reward_count[reset_bool] = 0
                terminations[reset_bool] = True
                infos['lives'][reset_bool] = 0
                obs, _ = self.env.reset(self.env_id_arr[reset_bool])
                print("Reset Info: env_id=", self.env_id_arr[reset_bool], "len(obs)=", len(obs))
                observations[reset_bool] = obs
            # switch terminations and truncated
            return truncated, [i/255.0 for i in observations], infos, terminations
        return truncated, [i/255.0 for i in observations], infos, terminations

    def reset(self, **kwargs):
        observations, info = self.envs.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.record_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        if self.max_no_reward_count is not None:
            self.max_no_reward_arr = np.ones(self.num_envs) * self.max_no_reward_count
        self.no_reward_count = np.zeros(self.num_envs)
        return [i/255.0 for i in observations], info

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.envs.step(action)
        # no_reward reset & float observations
        terminated, observations, infos, truncated = self._one_env_reset(rewards, terminated, observations, infos, truncated)

        self.episode_returns += rewards
        self.episode_lengths += 1
        # 上一次已经是0条命
        collect_bool = np.logical_and(infos['lives'] == 0, terminated)
        if collect_bool.sum() > 0:
            if self.num_envs == 1:
                infos['episode'] = {'r': self.episode_returns[0]}
            else:
                infos['final_info'] = [ 
                    {'episode': {'r': self.episode_returns[idx]}, 'env_id': idx} for idx, i in enumerate(infos['lives']) if  i == 0 
                ]
            self.record_episode_returns = self.episode_returns
            self.episode_returns[collect_bool] = 0.0 
            self.episode_lengths[collect_bool] = 0.0 
        return (
            observations,
            rewards,
            terminated, truncated,
            infos,
        )


class baseSkipFrame(gym.Wrapper):
    def __init__(
            self, 
            env, 
            skip: int, 
            cut_slices: List[List[int]]=None,
            start_skip: int=None,
            neg_action_kwargs: Dict=None,
            action_map: Dict = None,
            max_no_reward_count: Optional[int] = None,
            max_obs: bool = False
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
        self.action_map = action_map
        self.max_no_reward_count = max_no_reward_count
        self.no_reward_count = 0
        self.max_obs = max_obs
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    
    def _get_need_action(self, action):
        if self.action_map is None:
            return action 
        return self.action_map[action]
    
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
        
        action = self._get_need_action(action)
        total_reward = 0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            reward += neg_r
            done_f = terminated or truncated
            total_reward += reward
            if done_f:
                obs = self._obs_buffer.max(axis=0) if self.max_obs else obs
                obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
                return obs, total_reward, terminated, truncated, info

        # no reward max reset
        self.no_reward_count += 1
        if total_reward != 0:
            self.no_reward_count = 0
        if self.max_no_reward_count is not None and self.no_reward_count >= self.max_no_reward_count:
            print(f"Rest {self.max_no_reward_count=} {self.no_reward_count=}")
            self.no_reward_count = 0
            terminated = True

        obs = self._obs_buffer.max(axis=0) if self.max_obs else obs
        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        return obs, total_reward, terminated, truncated, info
    
    def _start_skip(self, seed=0, options=None, **kwargs):
        a = np.array([0.0, 0.0, 0.0]) if hasattr(self.env.action_space, 'low') else np.array(0) 
        for i in range(self.start_skip):
            obs, reward, terminated, truncated, info = self.env.step(a)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return obs, info

    def reset(self, seed=0, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        if self.start_skip is not None:
            obs, info = self._start_skip(seed, options, **kwargs)
        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        self.no_reward_count = 0
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
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            out_done = self.judge_out_of_route(obs)
            done_f = done or out_done
            reward = -10 if out_done else reward
            # reward = -100 if out_done else reward
            # reward = reward * 10 if reward > 0 else reward
            total_reward += reward
            tt_reward_list.append(reward)
            if done_f:
                break
        return obs[:84, 6:90, :], total_reward, done_f, done_f, info
    
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
    def __init__(self, env, resie_inner_area: bool=False):
        """RGP -> Gray
        (high, width, channel) -> (1, high, width) 
        """
        super().__init__(env)
        # change observation type for [ sync_vector_env ]
        self.resie_inner_area = resie_inner_area
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8 if resie_inner_area else np.float32
        )
    
    def observation(self, observation):
        if self.resie_inner_area:
            return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        tf = transforms.Grayscale()
        # channel first
        return tf(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape: int, resie_inner_area: bool=False):
        """reshape observe
        Args:
            env (_type_): _description_
            shape (int): reshape size  cv2.INTER_LINEAR  == BILINEAR  双线性插值
            resie_inner_area (bool): resie_inner_area resize 使用cv2.INTER_AREA 区域插值，适合缩小图像: 可以避免在缩小的图像中引入不期望的模糊，保持图像的锐度
        """
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.resie_inner_area = resie_inner_area
        # change observation type for [ sync_vector_env ]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        # print(f"{observation.max()=} {observation.min()=}")
        if self.resie_inner_area:
            return cv2.resize(
                observation, self.shape, interpolation=cv2.INTER_AREA
            ) / 255.0

        #  Normalize -> input[channel] - mean[channel]) / std[channel]
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


# reference: https://github.com/Stable-Baselines-Team/stable-baselines/blob/master/stable_baselines/common/atari_wrappers.py
# NoopResetEnv == baseSkipFrame
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        truncated = False
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            truncated = True
        self.lives = lives
        # test 继续，train的时候mask
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, reward, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
    

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.

        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward: (float)
        """
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.

        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(1)
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(2)
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


