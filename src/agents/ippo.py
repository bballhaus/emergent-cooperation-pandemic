"""Independent PPO. Per-agent selfish reward; per-agent critic on own observation + agent ID.

This is the "no enforced cooperation" baseline from Leibo et al. 2017 lifted to the
SEIR setting. If IPPO underperforms MAPPO and the peer-incentive condition closes the
gap, that's the positive finding the proposal is after.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..train.ppo_base import PPOBase, PPOConfig


class IPPOTrainer(PPOBase):
    name = "ippo"
    has_token_head = False

    def _critic_input_dim(self) -> int:
        return self.obs_dim + self.n_agents  # own obs + agent ID one-hot

    def _critic_input(self, obs: torch.Tensor, agent_ids: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(agent_ids, num_classes=self.n_agents).float()
        return torch.cat([obs, one_hot], dim=-1)

    def _reward_transform(self, rewards):
        return rewards  # selfish: each agent gets only its own reward
