"""Shared actor / critic networks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Dirichlet


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output_dim),
    )


@dataclass
class ActorOutput:
    allocation_dist: Dirichlet
    token_dist: Beta | None


class Actor(nn.Module):
    """Dirichlet allocation actor with optional token head."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        action_dim: int,
        hidden: int = 128,
        has_token_head: bool = False,
        local_init_bias: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.has_token_head = has_token_head
        self.input_dim = obs_dim + n_agents
        self.local_init_bias = local_init_bias if action_dim == n_agents else 0.0

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.alloc_head = nn.Linear(hidden, action_dim)
        if has_token_head:
            self.token_alpha = nn.Linear(hidden, 1)
            self.token_beta = nn.Linear(hidden, 1)

    def _featurize(self, obs: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(agent_id, num_classes=self.n_agents).float()
        return torch.cat([obs, one_hot], dim=-1)

    def forward(self, obs: torch.Tensor, agent_id: torch.Tensor) -> ActorOutput:
        z = self.trunk(self._featurize(obs, agent_id))
        logits = self.alloc_head(z)
        if self.local_init_bias != 0.0:
            local = F.one_hot(agent_id, num_classes=self.action_dim).float()
            logits = logits + self.local_init_bias * local
        alpha = F.softplus(logits) + 1.0
        alloc = Dirichlet(alpha)
        token = None
        if self.has_token_head:
            a = F.softplus(self.token_alpha(z)).squeeze(-1) + 1.0
            b = F.softplus(self.token_beta(z)).squeeze(-1) + 1.0
            token = Beta(a, b)
        return ActorOutput(allocation_dist=alloc, token_dist=token)


class Critic(nn.Module):
    """Scalar value critic."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
