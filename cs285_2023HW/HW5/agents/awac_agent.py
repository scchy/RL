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
        adv_norm: bool=False,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature
        self.critic_loss = nn.HuberLoss(delta=1.0)
        self.adv_norm = adv_norm
        print(f"{self.adv_norm=}")

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
            ac_dist = self.actor(next_observations.detach().clone())
            probs = ac_dist.probs  
            probs = torch.clamp_min(probs, 1e-3) # 最小概率 0.001
            probs = probs / probs.sum(-1, keepdim=True)   
            if self.use_double_q:
                next_q_main = self.critic(next_observations.detach().clone())       
                next_act = next_q_main.argmax(1, keepdim=True)
                next_qa_values = self.target_critic(next_observations.detach().clone())
                next_qs = next_qa_values.gather(1, next_act).squeeze(1) 
            else:
                next_qa_values = self.target_critic(next_observations)   # [B, A]
                # Expected-SARSA 
                next_qs = (probs * next_qa_values).sum(-1)
                #  最优动作 格（DDQN）
                # next_qs = next_qa_values.gather(1, probs.argmax(dim=1, keepdim=True)).squeeze(1)

            assert rewards.shape == next_qs.shape and dones.shape == next_qs.shape, f"{rewards.shape=} {next_qs.shape=} {dones.shape=}"
            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_qs * (1-dones)

        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.long().view(-1, 1)).view(-1)
        assert q_values.shape == target_values.shape, f"{next_qs.shape=} & {q_values.shape=} != {target_values.shape=}"
        loss = self.critic_loss(q_values, target_values.detach())
        with torch.no_grad():
            assert (q_values - target_values).abs().max() > 1e-5, \
                f"max(|q - target|) = {(q_values - target_values).abs().max().item():.6f} q == target"
        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
                "td_max": (q_values - target_values.detach()).abs().max().item()
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
        with torch.no_grad():  
            probs = action_dist.probs
            qa_values = self.critic(observations)
            q_values = qa_values.gather(1, actions.long().view(-1, 1)).view(-1)
            # V(s) = Σ_a π(a|s) * Q(s,a)  
            values = torch.sum(probs * qa_values, dim=-1).view(-1)
            # SAC V(s) = Σ_a π(a|s) * [ Q(s,a) - log π(a|s) ]  
            # logits = action_dist.logits                       # [B, A]
            # log_pi = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # 手动 log-softmax
            # values = torch.sum(probs * (qa_values - log_pi), dim=1)
            adv = q_values - values 
        return adv 

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        ac_dist = self.actor(observations)
        ac_log_prob = ac_dist.log_prob(actions)
        adv = self.compute_advantage(observations, actions, ac_dist)
        if self.adv_norm:
            adv = (adv - adv.mean())/(adv.std() + 1e-7)  
        # adv = adv - adv.max()
        w = torch.exp( adv/self.temperature ).detach() 
        assert adv.shape == ac_log_prob.shape, f"{w.shape=} != {ac_log_prob.shape=}"
        loss = -(ac_log_prob * w).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actor_optimizer.step()
        return {
            'actor_loss': loss.item(),
            'adv': w.detach().mean().item(),
            'entropy': ac_dist.entropy().mean().item(),
        }

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
        actor_metrics = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_metrics["actor_loss"]
        metrics["actor_adv"] = actor_metrics["adv"]
        metrics["entropy"] = actor_metrics["entropy"]
        metrics["UStep"] = step // self.target_update_period
        return metrics
