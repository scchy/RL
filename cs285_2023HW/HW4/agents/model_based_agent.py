# python 3
# Create Date: 2025-08-27
# reference: https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw4/cs285/agents/model_based_agent.py
# =================================================================================================================

from typing import Callable, Optional, Tuple
import numpy as np
from torch import nn
import torch
import gymnasium as gym
from utils.utools import (
    from_numpy, to_numpy, device
)

class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env, 
        make_dynamics_model: Callable[[Tuple[int, ...]], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int, 
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        ).to(device)
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=device)
        )
        self.epsilon = 1e-8
        self.cnt = 1e-8

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = from_numpy(obs)
        acs = from_numpy(acs)
        next_obs = from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        norm_in = self.norm_func(torch.cat([obs, acs], dim=-1), tp='obs_acs')
        hat_delta = self.dynamics_models[i]( 
            norm_in
        )
        delta =  self.norm_func(next_obs - obs, tp='obs_delta')
        loss = self.loss_fn(delta, hat_delta)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_numpy(loss)

    @torch.no_grad()
    def norm_func(self, input_tensor, tp='obs_acs', reverse=False):
        assert tp in ('obs_acs', 'obs_delta'), f"Please use ('obs_acs', 'obs_delta') for tp, but input {tp=}"
        if tp == 'obs_acs':
            if reverse:
                return input_tensor * (self.epsilon + self.obs_acs_std) + self.obs_acs_mean
            return (input_tensor - self.obs_acs_mean) / (self.epsilon + self.obs_acs_std)

        if reverse:
            return input_tensor * (self.epsilon + self.obs_delta_std) + self.obs_delta_mean
        return (input_tensor - self.obs_delta_mean) / (self.epsilon + self.obs_delta_std)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = from_numpy(obs)
        acs = from_numpy(acs)
        obs_acs = torch.cat([obs, acs], dim=1)
        next_obs = from_numpy(next_obs)
        delta = next_obs - obs
        # TODO(student): update the statistics
        m_, v_, tt_cnt = self.update_mean_var_count_from_moments(
            self.obs_acs_mean, torch.square(self.obs_acs_std), self.cnt, 
            obs_acs.mean(axis=0),  obs_acs.var(axis=0), obs_acs.shape[0]
        )
        self.obs_acs_mean = m_
        self.obs_acs_std = torch.sqrt(v_)

        m_, v_, tt_cnt = self.update_mean_var_count_from_moments(
            self.obs_delta_mean, torch.square(self.obs_delta_std), self.cnt, 
            delta.mean(axis=0),  delta.var(axis=0), delta.shape[0]
        )
        self.obs_delta_mean = m_
        self.obs_delta_std = torch.sqrt(v_)
        self.cnt = tt_cnt
    
    def update_mean_var_count_from_moments(
        self, m, v, cnt, batch_m, batch_v, batch_cnt
    ):
        # reference: gymnasium.wrappers.normalize
        # print(f'{batch_m.device=} {m.device=} {device=}')
        delta = batch_m - m
        tot_cnt = cnt + batch_cnt

        new_m = m + delta * batch_cnt / tot_cnt
        m_a = v * cnt 
        m_b = batch_v * batch_cnt
        M2 = m_a + m_b + torch.square(delta) * cnt * batch_cnt / tot_cnt
        new_v = M2 / tot_cnt
        new_cnt = tot_cnt
        return new_m, new_v, new_cnt 

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = from_numpy(obs)
        acs = from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        hat_delta = self.dynamics_models[i](self.norm_func(torch.cat([obs, acs], dim=-1), tp='obs_acs'))
        pred_next_obs = obs + self.norm_func(hat_delta, tp='obs_delta', reverse=True)
        return to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        for acs in action_sequences.transpose(1, 0, 2) :
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            next_obs = np.stack([self.get_dynamics_predictions(i, obs[i, ...], acs) for i in range(self.ensemble_size)], axis=0)
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.

            rewards = np.stack([
              self.env.get_reward(next_obs[i], acs)[0] for i in range(self.ensemble_size)], axis=0)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences), f"{rewards.shape=} != ({self.ensemble_size}, {self.mpc_num_action_sequences})"

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)


    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
                # Step1: i==0 → 用初始 mean/ std
                if i == 0:
                    mean_ = (self.env.action_space.high + self.env.action_space.low) / 2.0
                    std_  = (self.env.action_space.high - self.env.action_space.low) / 2.0
                else:
                    mean_, std_ = elite_mean, elite_std
                # Step2: 采样
                pop = np.random.normal(
                    loc=mean_,
                    scale=std_,
                    size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim)
                ).clip(self.env.action_space.low, self.env.action_space.high)

                # Step3: 评估  
                scores = self.evaluate_action_sequences(obs, pop) 

                # Step4: 重估 → 精英样本的 mean / std 作为下一轮参数
                elite_idx = np.argpartition(scores, self.cem_num_elite)[:self.cem_num_elite] # 取最大的n个元素索引（不排序）
                elite_pop = pop[elite_idx]
                elite_mean = elite_pop.mean(dim=0)
                elite_std  = elite_pop.std(dim=0) + 1e-8    
            return elite_mean[0]
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")




















