

from typing import Sequence, Callable, Tuple, Optional
import torch
from torch import nn
import numpy as np
from utils.utools import from_numpy, to_numpy


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
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.gamma = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()
        self.training = True
        self.update_target_critic()
    
    def train(self):
        self.critic.train()
        self.training = True
    
    def eval(self):
        self.critic.eval()
        self.training = False

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if self.training and np.random.random() < epsilon:
            action = np.random.randint(self.num_actions)
            return action
        
        action = self.critic(observation)
        action = action.argmax(dim=-1)
        return to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values = self.critic(next_obs)

            if self.use_double_q:
                next_qa_star_values = self.target_critic(next_obs)
                next_action = next_qa_star_values.argmax(dim=-1).view(-1, 1)
            else:
                next_action = next_qa_values.argmax(dim=-1).view(-1, 1)
            
            next_q_values = next_qa_values.gather(1, next_action).view(-1)
            target_values = reward + self.gamma * next_q_values * (1 - done)
            

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = qa_values.gather(1, action.long().view(-1, 1)).view(-1)
        # print(f"{next_q_values.shape=} {reward.shape=} {done.shape=} {target_values.shape=} {q_values.shape=}")
        loss = self.critic_loss(target_values, q_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()
        self.lr_scheduler.step()
        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

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
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(
            obs,
            action,
            reward,
            next_obs,
            done,
        )
        if self.use_double_q and step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats

