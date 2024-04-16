import gymnasium as gym
import numpy as np

def make_env(env_id, obs_norm_trans_flag=False, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        if obs_norm_trans_flag:
            env = gym.wrappers.NormalizeObservation(gym.wrappers.ClipAction(env))
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
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
