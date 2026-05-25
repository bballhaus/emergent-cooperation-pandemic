"""IPPO + peer incentives — the novel contribution.

Each agent keeps a per-episode token budget (default 100). When agent i receives
transfers this step (deliveries that arrived today), the actor emits a token *fraction*
in [0, 1] from a Beta head; tokens_emitted = floor(token_frac * budget_remaining_i),
distributed to senders in proportion to how much each sender contributed. Each token
converts to a small reward bonus for the sender at fixed exchange rate.

Design points relative to MATE (Phan et al. 2024):
  - Continuous allocation actions, not discrete moves
  - Asynchronous demand shocks → long lag between giving and needing
  - Fixed per-episode budget → built-in defense against collusive token inflation;
    we monitor `token_to_transfer_ratio` across training to detect collusion early

Token bonus arrives at the receiver's ack step (one step after the original send) and
is credited to the sender's reward at that step. GAE propagates this delayed signal
back to the send-time action with negligible discount cost (γ^1 ≈ 0.99).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .ippo import IPPOTrainer
from ..train.ppo_base import PPOConfig


@dataclass
class PeerIncentiveConfig(PPOConfig):
    token_budget: int = 100        # per-agent tokens available per episode
    token_exchange_rate: float = 1e-4   # reward bonus per token (in env's reward units)
    log_token_stats_every: int = 10


class PeerIncentiveTrainer(IPPOTrainer):
    name = "ippo_peer"
    has_token_head = True

    def __init__(self, env, config: PeerIncentiveConfig):
        super().__init__(env, config)
        self.pi_cfg: PeerIncentiveConfig = config
        # Tracking across an episode.
        self._budget: np.ndarray = np.full(self.n_agents, self.pi_cfg.token_budget, dtype=np.int64)
        # token_to_transfer_ratio: cumulative tokens emitted / cumulative ventilators received,
        # tracked across the whole training run to monitor for collusion (ratio drifting up
        # without matching welfare gains => agents are inflating tokens, not actually sharing).
        self._cum_tokens_emitted = 0
        self._cum_transfers_received = 0
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._budget = np.full(self.n_agents, self.pi_cfg.token_budget, dtype=np.int64)

    def collect_rollout(self, buffer) -> dict:
        # Hook in episode-state reset before the base class kicks off the env reset.
        self._reset_episode_state()
        ep = super().collect_rollout(buffer)
        # Attach peer-incentive diagnostics.
        ep["tokens_emitted_cumulative"] = int(self._cum_tokens_emitted)
        ep["transfers_received_cumulative"] = int(self._cum_transfers_received)
        ratio = (
            self._cum_tokens_emitted / max(self._cum_transfers_received, 1)
        )
        ep["token_to_transfer_ratio"] = float(ratio)
        return ep

    def _maybe_shape_reward(
        self,
        r_raw: np.ndarray,
        infos: dict,
        agents: list[str],
        extras: dict,
    ) -> np.ndarray:
        """Apply token bonuses to senders based on receivers' ack-step token emissions."""
        r = r_raw.copy()
        token_fracs = extras.get("token_frac")
        if token_fracs is None:
            return r

        # For each agent i (the potential ack-er):
        for i, agent in enumerate(agents):
            received = infos[agent].get("transfers_received", {})
            if not received:
                continue
            total_received = sum(received.values())
            self._cum_transfers_received += total_received
            budget = int(self._budget[i])
            if budget <= 0:
                continue
            tokens_to_emit = int(np.floor(float(token_fracs[i]) * budget))
            tokens_to_emit = min(tokens_to_emit, budget)
            if tokens_to_emit <= 0:
                continue

            # Distribute proportional to sender contribution.
            for sender_id, amount in received.items():
                share = amount / max(total_received, 1)
                allocated = int(round(tokens_to_emit * share))
                if allocated <= 0:
                    continue
                r[sender_id] += self.pi_cfg.token_exchange_rate * allocated * self.env.config.reward_scale
                self._cum_tokens_emitted += allocated
            self._budget[i] -= tokens_to_emit
        return r
