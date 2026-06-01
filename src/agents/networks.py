"""Shared actor / critic networks.

The actor outputs Dirichlet concentration parameters over the simplex of size n_cities;
this is the natural distribution on the simplex action space the env expects (entry i is
"fraction of stockpile to use locally", entries j != i are transfer fractions to city j).

The critic is reused by both IPPO and MAPPO. The difference is in `input_dim`:
  - IPPO: critic_input_dim = obs_dim + n_agents  (own observation + agent ID one-hot)
  - MAPPO: critic_input_dim = obs_dim * n_agents  (concatenated global state)

Parameter sharing across agents is the standard MARL trick (Yu et al. 2022 §3.2): one
network for all agents, agent identity injected via one-hot append. This is sound here
because the action structure is identical across agents even though populations and
hospital capacities differ.

For the peer-incentive condition, the actor exposes an optional Beta head emitting a
scalar token fraction in [0, 1], multiplied by remaining budget by the trainer.
"""

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
    token_dist: Beta | None     # None unless the actor was built with has_token_head=True


class Actor(nn.Module):
    """Outputs Dirichlet over allocation simplex; optionally a Beta for token fraction."""

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
        # Additive prior on the agent's own allocation slot (index == agent id, since
        # action_dim == n_agents here: slot i is "use locally", slots j!=i are transfers).
        # Without it the Dirichlet inits near-uniform, so agents blindly transfer ~(n-1)/n of
        # their stockpile every step (milestone: 9M transfers, deaths worse than the heuristic).
        # It only shifts the init; alloc_head can learn negative logits to transfer more.
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
        # +1 shifts concentration above 1, keeping the distribution well-behaved near uniform
        # at init (avoids collapsing to a corner of the simplex with infinite log-prob spikes).
        alpha = F.softplus(logits) + 1.0
        alloc = Dirichlet(alpha)
        token = None
        if self.has_token_head:
            a = F.softplus(self.token_alpha(z)).squeeze(-1) + 1.0
            b = F.softplus(self.token_beta(z)).squeeze(-1) + 1.0
            token = Beta(a, b)
        return ActorOutput(allocation_dist=alloc, token_dist=token)


class Critic(nn.Module):
    """Scalar value head. Reused for IPPO (own-obs input) and MAPPO (global-state input)."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
