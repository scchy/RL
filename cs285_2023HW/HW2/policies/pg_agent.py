# Python3
# Preference:
#       from cs285.agents.pg_agent import PGAgent
#       
# ===========================================================================
from typing import Optional, Sequence
import numpy as np
import torch
from torch.nn import Module
from torch import nn
from .utils import to_numpy, from_numpy
from .base_net import MLPPolicyPG, ValueCritic


class PGAgent:
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs)
        rewards = np.concatenate(rewards)  
        actions = np.concatenate(actions)  
        q_values = np.concatenate(q_values)  
        terminals = np.concatenate(terminals)  
        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages) 

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(r_arr) for r_arr in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(r_arr) for r_arr in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            # use return-go or discount reward  
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = self.critic(from_numpy(obs)).detach().cpu().numpy().flatten()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)
                gae = 0
                for t in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    mask = 1 - terminals[t] * 1
                    delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
                    advantages[t] = gae = delta + self.gae_lambda * self.gamma * gae * mask
                # remove dummy advantage
                advantages = advantages[:-1]
                # print('gae adv=', advantages)

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one
        # within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-5)
        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        bf = rewards[0]
        for t, r_i in enumerate(rewards[1:]):
            bf += self.gamma ** (t+1) * r_i
        return np.array([bf] * len(rewards))

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        只关注后续
        """
        lst = rewards[-1]
        rt = [lst]
        for t, r_i in enumerate(rewards[:-1][::-1]):
            lst = r_i + self.gamma * lst
            rt.insert(0, lst)
        return np.array(rt)
