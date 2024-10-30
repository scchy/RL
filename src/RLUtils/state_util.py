import gymnasium as gym
import numpy as np
import cloudpickle
import envpool
from .env_wrapper import (
    baseSkipFrame, EpisodicLifeEnv, ClipRewardEnv, FireResetEnv, 
    ResizeObservation, GrayScaleObservation, 
    envPoolRecordEpisodeStatistics, 
    spSyncVectorEnv
)


def make_env(env_id, obs_norm_trans_flag=False, reward_norm_trans_flag=False, gamma=0.99, render_mode=None, **kwargs):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if obs_norm_trans_flag:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if reward_norm_trans_flag:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def save_env(env, file_path):
    # 保存环境状态
    with open(file_path, 'wb') as file:
        cloudpickle.dump(env, file)
    print(f'saved env -> {file_path}')

    # # 加载环境状态
    # with open('env_state.pkl', 'rb') as file:
    #     env_loaded = cloudpickle.load(file)


def make_atari_env(env_id, episod_life=False, clip_reward=True, action_map=None, 
                   skip=5, 
                   start_skip=30, 
                   ppo_train=False, 
                   max_no_reward_count=None,
                   resize_inner_area=False,
                   fire_flag=True,
                   max_obs=False,
                   **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = baseSkipFrame(env, skip=skip, start_skip=start_skip, 
                            action_map=action_map,
                            max_no_reward_count=max_no_reward_count,
                            max_obs=max_obs)
        if episod_life:
            env = EpisodicLifeEnv(env)
        if ("FIRE" in env.unwrapped.get_action_meanings()) and fire_flag:
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        env = ResizeObservation(GrayScaleObservation(env, resie_inner_area=resize_inner_area), shape=84, resie_inner_area=resize_inner_area)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def make_envpool_atria(
        env_id, num_envs, 
        seed=42, 
        img_height=84, 
        img_width=84,
        stack_num=4,
        gray_scale=True,
        noop_max=30,
        episodic_life=True, 
        reward_clip=True, 
        frame_skip=4,
        max_episode_steps=27000,
        max_no_reward_count=None
    ):
    """doc: https://envpool.readthedocs.io/en/latest/env/atari.html
        task_id (str): see available tasks below;
        num_envs (int): how many environments you would like to create;
        batch_size (int): the expected batch size for return result, default to num_envs;
        num_threads (int): the maximum thread number for executing the actual env.step, default to batch_size;
        seed (int): the environment seed, default to 42;
        max_episode_steps (int): the maximum number of steps for one episode, default to 27000, which corresponds to 108000 frames or roughly 30 minutes of game-play (Hessel et al. 2018, Table 3) because of the 4 skipped frames;
        img_height (int): the desired observation image height, default to 84;
        img_width (int): the desired observation image width, default to 84;
        stack_num (int): the number of frames to stack for a single observation, default to 4;
        gray_scale (bool): whether to use gray scale env wrapper, default to True;
        frame_skip (int): the number of frames to execute one repeated action, only the last frame would be kept, default to 4;
        noop_max (int): the maximum number of no-op action being executed when calling a single env.reset, default to 30;
        episodic_life (bool): make end-of-life == end-of-episode, but only reset on true game over. It helps the value estimation. Default to False;
        zero_discount_on_life_loss (bool): when the agent losses a life, the discount in dm_env.TimeStep is set to 0. This option doesn’t affect gym’s behavior (since there is no discount field in gym’s API). Default to False;
        reward_clip (bool): whether to change the reward to sign(reward), default to False;
        repeat_action_probability (float): the action repeat probability in ALE configuration, default to 0 (no action repeat to perform deterministic result);
        use_inter_area_resize (bool): whether to use cv::INTER_AREA for image resize, default to True.
        use_fire_reset (bool): whether to use fire-reset wrapper, default to True.
        full_action_space (bool): whether to use full action space of ALE of 18 actions, default to False.
    """
    envs = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=seed,
        max_episode_steps=max_episode_steps
    )
    envs.num_envs = num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = envPoolRecordEpisodeStatistics(envs, max_no_reward_count=max_no_reward_count)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return envs


def Pendulum_dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    action_range = action_upbound - action_lowbound
    return action_lowbound + ( discrete_action / (action_dim - 1) ) * action_range


def gym_env_desc(env_name):
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    try:
        action_shape = env.action_space.n
        action_type = '离散'
        extra=''
    except Exception as e:
        action_shape = env.action_space.shape
        low_ = env.action_space.low[0]  # 连续动作的最小值
        up_ = env.action_space.high[0]  # 连续动作的最大值
        extra=f'<{low_} -> {up_}>'
        action_type = '连续'
    print(f'[ {env_name} ](state: {state_shape},action: {action_shape}({action_type} {extra}))')
    return 
