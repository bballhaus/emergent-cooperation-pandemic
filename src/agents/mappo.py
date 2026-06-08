"""MAPPO with centralized critic and team reward."""

from __future__ import annotations

import numpy as np
import torch

from ..train.ppo_base import PPOBase, PPOConfig


class MAPPOTrainer(PPOBase):
    name = "mappo"
    has_token_head = False

    def _critic_input_dim(self) -> int:
        return self.env.global_state_dim()

    def _critic_input(self, obs: torch.Tensor, agent_ids: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        return global_state

    def _reward_transform(self, rewards: np.ndarray) -> np.ndarray:
        team = rewards.sum()
        return np.full_like(rewards, team, dtype=np.float32)
