import gymnasium as gym
import numpy as np
import cloudpickle
from .env_wrapper import baseSkipFrame, EpisodicLifeEnv, ClipRewardEnv, FireResetEnv, ResizeObservation, GrayScaleObservation


def make_env(env_id, obs_norm_trans_flag=False, reward_norm_trans_flag=False, gamma=0.99, render_mode=None, **kwargs):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if obs_norm_trans_flag:
            env = gym.wrappers.NormalizeObservation(gym.wrappers.ClipAction(env))
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


def make_atari_env(env_id, episod_life=True, clip_reward=True, action_map=None, skip=5, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = baseSkipFrame(env, skip=skip, start_skip=30, action_map=action_map)
        if episod_life:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        env = ResizeObservation(GrayScaleObservation(env), shape=84)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


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
