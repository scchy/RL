# python3
# Create Date: 2025-09-01
# Author: Chengchao.Sun 
# ========================================================================================

import numpy as np 
import gymnasium as gym 
from gymnasium.spaces import Box 
import matplotlib.pyplot as plt 


class Obstacles(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }
    def __init__(
        self,
        start=[-0.5, 0.75], 
        end=[0.7, -0.8], 
        random_starts=True, **kwargs
    ):
        super(Obstacles, self).__init__()
        self.plt = plt
        self.action_dim = self.ac_dim = 2
        self.observation_dim = self.obs_dim = 4
        self.boundary_min = -0.99
        self.boundary_max = 0.99
        low = self.boundary_min * np.ones((self.action_dim,))
        high = self.boundary_max * np.ones((self.action_dim,))
        self.action_space = Box(low, high, dtype=np.float32)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high, dtype=np.float32)

        self.env_name = "obstacles"
        self.is_gym = True

        self.start = np.array(start)
        self.end = np.array(end)
        self.current = np.array(start)
        self.random_starts = random_starts

        # obstacles are rectangles, specified by [x of top left, y of topleft, width x, height y]
        self.obstacles = []
        self.obstacles.append([-0.4, 0.8, 0.4, 0.3])
        self.obstacles.append([-0.9, 0.3, 0.2, 0.6])
        self.obstacles.append([0.6, -0.1, 0.12, 0.4])
        self.obstacles.append([-0.1, 0.2, 0.15, 0.4])
        self.obstacles.append([0.1, -0.7, 0.3, 0.15])

        self.eps = 0.1
        self.fig = self.plt.figure()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def seed(self, seed):
        np.random.seed(seed)
    
    def pick_start_pos(self):
        if self.random_starts:
            temp = np.random.uniform(
                [self.boundary_min, self.boundary_min + 1.25],
                [self.boundary_max - 0.4, self.boundary_max],
                (self.action_dim,),
            )
            if not self.is_valid(temp[None, :]):
                temp = self.pick_start_pos()
        else:
            temp = self.start
        return temp

    def reset(self, seed=None, *args, **kwargs):
        if seed:
            self.seed(seed)

        self.reset_pose = self.pick_start_pos()
        self.reset_vel = self.end
        ob = self.do_reset(self.reset_pose, self.reset_vel)
        return ob.astype(np.float32), {"ob": ob}

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):
        self.current = reset_pose.copy()
        self.end = reset_vel.copy()

        # clear
        self.counter = 0
        self.plt.clf()

        # return
        return self.get_obs() 

    def get_obs(self):
        return np.concatenate([self.current, self.end])

    def get_score(self, obs):
        curr_pos = obs[:2]
        end_pos = obs[-2:]
        score = -1 * np.abs(curr_pos - end_pos)
        return score

    def get_reward(self, obs, actions):
        """
        Parameters:
        --------
        :param np.array obs: (..., obs_dim)  前 2 维是 curr_pos, 后 2 维是 end_pos
        :param np.array act: (..., ac_dim)   
        return:
        ----
         (reward, done) 与输入同维度
        """
        obs = np.atleast_2d(obs)
        curr, end = obs[:, :2], obs[:, -2:]
        dist = np.linalg.norm(curr - end, axis=1)

        # reward: neg dis 
        reward = -dist
        # done: goal or out of bound 
        reached = dist < self.eps 
        oob = (
            (curr < self.boundary_min) |
            (curr > self.boundary_max)
        ).any(axis=1)
        done = reached | oob
        
        if obs.shape[0] <= 1:
            return reward.item(), done.item()
        return reward, done 
    
    def step(self, action):
        self.counter += 1
        action = np.clip(action, -1, 1)  # clip (-1, 1)
        action = action / 10.0  # scale (-1,1) to (-0.1, 0.1)

        temp = self.current + action
        if self.is_valid(temp[None, :]):
            self.current = temp

        ob = self.get_obs().astype(np.float32)
        reward, terminated = self.get_reward(ob, action)
        score = self.get_score(ob)
        env_info = {"ob": ob, "rewards": reward, "score": score}
        truncated = self.counter >= self.max_episode_steps
        return ob, reward, terminated, truncated, env_info

    def render(self, mode=None):
        if mode != 'human':
            return
        # boundaries
        self.plt.plot(
            [self.boundary_min, self.boundary_min],
            [self.boundary_min, self.boundary_max],
            "k",
        )
        self.plt.plot(
            [self.boundary_max, self.boundary_max],
            [self.boundary_min, self.boundary_max],
            "k",
        )
        self.plt.plot(
            [self.boundary_min, self.boundary_max],
            [self.boundary_min, self.boundary_min],
            "k",
        )
        self.plt.plot(
            [self.boundary_min, self.boundary_max],
            [self.boundary_max, self.boundary_max],
            "k",
        )
        # obstacles
        for obstacle in self.obstacles:
            tl_x = obstacle[0]
            tl_y = obstacle[1]
            tr_x = tl_x + obstacle[2]
            tr_y = tl_y
            bl_x = tl_x
            bl_y = tl_y - obstacle[3]
            br_x = tr_x
            br_y = bl_y
            self.plt.plot([bl_x, br_x], [bl_y, br_y], "r")
            self.plt.plot([tl_x, tr_x], [tl_y, tr_y], "r")
            self.plt.plot([bl_x, bl_x], [bl_y, tl_y], "r")
            self.plt.plot([br_x, br_x], [br_y, tr_y], "r")
        # current and end
        self.plt.plot(self.end[0], self.end[1], "go")
        self.plt.plot(self.current[0], self.current[1], "ko")
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return [img]

    def is_valid(self, dat):
        oob_mask = np.any(self.oob(dat), axis=1)

        # old way
        self.a = self.boundary_min + (self.boundary_max - self.boundary_min) / 3.0
        self.b = self.boundary_min + 2 * (self.boundary_max - self.boundary_min) / 3.0
        data_mask = (
            (dat[:, 0] < self.a)
            | (dat[:, 0] > self.b)
            | (dat[:, 1] < self.a)
            | (dat[:, 1] > self.b)
        )

        #
        in_obstacle = False
        for obstacle in self.obstacles:
            tl_x = obstacle[0]
            tl_y = obstacle[1]
            tr_x = tl_x + obstacle[2]
            tr_y = tl_y
            bl_x = tl_x
            bl_y = tl_y - obstacle[3]
            br_x = tr_x
            br_y = bl_y

            if (
                dat[:, 0] > tl_x
                and dat[:, 0] < tr_x
                and dat[:, 1] > bl_y
                and dat[:, 1] < tl_y
            ):
                in_obstacle = True
                return False

        # not in obstacle, so return whether or not its in bounds
        return not oob_mask

    def oob(self, x):
        return (x <= self.boundary_min) | (x >= self.boundary_max)

