# python3
# Create Date: 2025-08-29
# Author: Chengchao.Sun 
# ========================================================================================

import numpy as np 
from gymnasium import utils 
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.spaces import Box 


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }
    LEG_RANGE   = 0.2
    SHIN_RANGE  = 0.0
    FOOT_RANGE  = 0.0
    PENALTY     = 10.0

    def __init__(self, **kwargs):
        super().__init__(
            model_path="half_cheetah.xml",
            frame_skip=1,
            observation_space=Box(-np.inf, np.inf, (21,), np.float64),
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.skip = self.frame_skip

        self.action_dim = self.ac_dim = self.action_space.shape[0]
        self.observation_dim = self.obs_dim = self.observation_space.shape[0]

    def get_reward(self, obs, act):
        """get reward/s of given (obs, act) datapoint or datapoints

        Args:
            obs: (batchsize, obs_dim) or (obs_dim,)
            act: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """
        # initialize and reshape as needed, for batch mode
        obs = obs[None] if obs.ndim == 1 else obs
        act = act[None] if act.ndim == 1 else act

        self.reward_dict = {}

        xvel       = obs[:, 9]
        leg, shin, foot = obs[:, 6], obs[:, 7], obs[:, 8]
        body_angle = obs[:, 2]
 
        # calc rew
        # 布尔掩码 → 0/1 → 乘惩罚
        leg_pen  = (leg  > self.LEG_RANGE)  * -self.PENALTY
        shin_pen = (shin > self.SHIN_RANGE) * -self.PENALTY
        foot_pen = (foot > self.FOOT_RANGE) * -self.PENALTY

        reward = xvel + leg_pen + shin_pen + foot_pen
        done   = np.zeros(obs.shape[0], dtype=bool)
        return (reward.item(), done.item()) if obs.shape[0] == 1 else (reward, done)

    def get_score(self, obs):
        xposafter = obs[0]
        return xposafter

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        reward, terminated = self.get_reward(obs, action)
        info = {
            "obs_dict": {"x": obs[0], "xvel": obs[9]},
            "score": obs[0],           # 前进距离
        }
        return obs, reward, terminated, False, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, self.model.nv)
        self.set_state(qpos, qvel)
        return self.get_obs()

    def get_obs(self):
        self.obs_dict = {}
        self.obs_dict["joints_pos"] = self.data.qpos.flat.copy()
        self.obs_dict["joints_vel"] = self.data.qvel.flat.copy()
        self.obs_dict["com_torso"] = self.get_body_com("torso").flat.copy()

        return np.concatenate(
            [
                self.obs_dict["joints_pos"],  # 9
                self.obs_dict["joints_vel"],  # 9
                self.obs_dict["com_torso"],  # 3
            ]
        )

    def render(self):
        ren = super().render()
        self.renderer.render_step()
        return ren
