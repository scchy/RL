# python3
# Create Date: 2025-09-15
# Author: Scc_hy
# Fun: AWAC 
# reference:
# ============================================================================================

import torch
from torch import nn
import numpy as np
from typing import Callable, Optional, Tuple, Sequence
from .dqn_agent import DQNAgent
from utils.utools import (
    from_numpy, to_numpy, device
)


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_qa_values = self.critic(next_observations)

            # Use the actor to compute a critic backup
            ac_dist = self.actor(next_observations)
            next_qs = next_qa_values.gather(1, ac_dist.sample().view(-1, 1))

            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_qs * (1-dones)

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.long().view(-1, 1)).view(-1)
        assert q_values.shape == target_values.shape
        loss = self.critic_loss(q_values, target_values.detach())
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

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.view(-1, 1))
        # V(s) = Σ_a π(a|s) * Q(s,a)  
        values = torch.sum(action_dist.probs * qa_values, dim=-1) 
        # SAC V(s) = Σ_a π(a|s) * [ Q(s,a) - log π(a|s) ]  
        # log_pi = action_dist.log_prob(torch.arange(action_dist.logits.shape[-1], device=observations.device))
        # values = torch.sum(action_dist.probs * (qa_values - log_pi), dim=-1) 
        advantages = q_values - values 
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        ac_dist = self.actor(observations)
        ac_log_prob = ac_dist.log_prob(actions)
        adv = self.compute_advantage(observations, actions, ac_dist)
        w = torch.softmax(adv/self.temperature)
        loss = -(ac_log_prob * w.detach()).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(
            self, 
            observations: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            next_observations: torch.Tensor,
            dones: torch.Tensor, 
            step: int
    ):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
