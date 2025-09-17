# python3
# Create Date: 2025-09-15
# Author: Scc_hy
# Fun: IQL 
# paper: https://arxiv.org/abs/1810.12894
# ========================================================================

from typing import Optional, Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
from .awac_agent import AWACAgent


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_value_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.value_critic = make_value_critic(observation_shape)
        self.target_value_critic = make_value_critic(observation_shape)
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        self.expectile = expectile
        
    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): Compute advantage with IQL
        qa_values = self.critic(observations)
        q_value = qa_values.gather(1, actions.view(-1, 1)).view(-1, 1)
        value = torch.sum(action_dist.probs * qa_values, dim=-1)
        adv = q_value - value
        return adv

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        Update Q(s, a)
        """
        # TODO(student): Update Q(s, a) to match targets (based on V)
        with torch.no_grad():
            target_values = rewards + self.value_critic(next_observations) * (1 - dones)

        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.view(-1, 1)).view(-1, 1)
        loss = self.critic_loss(q_values, target_values.detach())

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": self.critic_loss(q_values, target_values).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ):
        """
        Compute the expectile loss for IQL 
        - expectile=0.5 退化为普通 MSE。
        - expectile→1 时只对“Q>V”部分惩罚，实现 IQL 的“乐观”更新。
        """
        # TODO(student): Compute the expectile loss
        adv = target_qs.detach() - vs
        # |\tau - \mathbb{1}\{\mu \le 0 \}|\mu ^2
        # weight = torch.where(adv < 0,  expectile - 1, expectile).abs()
        weight = torch.where(adv < 0,  1 - expectile, expectile) 
        loss = (weight * adv.pow(2)).mean()
        return loss

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # TODO(student): Compute target values for V(s)
        vs = self.value_critic(observations)
        target_qa = self.critic(observations)
        target_values = target_qa.gather(1, actions.view(-1, 1)).view(-1, 1)

        # TODO(student): Update V(s) using the loss from the IQL paper
        loss = self.iql_expectile_loss(self.expectile, vs, target_values)

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (vs - target_values).mean().item(),
            "vs": vs.mean().item(),
            "target_values": target_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }


    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(observations, actions, rewards, next_observations, dones)
        metrics["actor_loss"] = self.update_actor(observations, actions)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()
        
        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
