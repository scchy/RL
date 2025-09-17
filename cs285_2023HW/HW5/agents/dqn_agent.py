# python3
# Create Date: 2025-09-11
# Author: Scc_hy
# Fun: DQN 
# reference: 
# ============================================================================================

from typing import Sequence, Callable, Tuple, Optional
import torch
from torch import nn
import numpy as np
from utils.utools import (
    from_numpy, to_numpy, device
)


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int, 
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False, 
        clip_grad_norm: Optional[float] = None,
    ):
        super(DQNAgent, self).__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0
    ):
        """
        Used for evaluation.
        """
        obs = from_numpy(np.asarray(obs))[None]
        if epsilon > np.random.random():
            action = np.random.randint(self.num_actions)
            return action
        qa = self.critic(obs)
        return qa.argmax().detach().cpu().numpy()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ): # -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """
        batch_size = obs.size(0)
        # TODO(student): paste in your code from HW3, and make sure the return values exist
        with torch.no_grad():
            next_qa_values = self.critic(next_obs)

            if self.use_double_q:
                next_star_qa_values = self.traget_critic(next_obs)
                next_action = next_star_qa_values.argmax(dim=-1).view(-1, 1)
            else:
                next_action = next_qa_values.argmax(dim=-1).view(-1, 1)

            next_q_values = next_qa_values.gather(1, next_action).view(-1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount * next_q_values * (1 - done)
            assert target_values.shape == (batch_size,), target_values.shape

        qa_values = self.critic(obs)
        q_values = qa_values.gather(1, action.long().view(-1, 1)).view(-1)
        loss = self.critic_loss(q_values, target_values)
        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()
        self.lr_scheduler.step()
        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(
            obs,
            action,
            reward,
            next_obs,
            done
        )

        if self.use_double_q and step % self.target_update_period == 0:
            self.update_target_critic()

        return metrics
