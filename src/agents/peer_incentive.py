"""IPPO with peer incentive tokens."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .ippo import IPPOTrainer
from ..train.ppo_base import PPOConfig


@dataclass
class PeerIncentiveConfig(PPOConfig):
    token_budget: int = 100
    token_exchange_rate: float = 5e-3
    log_token_stats_every: int = 10


class PeerIncentiveTrainer(IPPOTrainer):
    name = "ippo_peer"
    has_token_head = True

    def __init__(self, env, config: PeerIncentiveConfig):
        super().__init__(env, config)
        self.pi_cfg: PeerIncentiveConfig = config
        self._budget: np.ndarray = np.full(self.n_agents, self.pi_cfg.token_budget, dtype=np.int64)
        self._cum_tokens_emitted = 0
        self._cum_transfers_received = 0
        self._per_city_emitted = np.zeros(self.n_agents, dtype=np.int64)
        self._per_city_received = np.zeros(self.n_agents, dtype=np.int64)
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._budget = np.full(self.n_agents, self.pi_cfg.token_budget, dtype=np.int64)
        self._per_city_emitted = np.zeros(self.n_agents, dtype=np.int64)
        self._per_city_received = np.zeros(self.n_agents, dtype=np.int64)

    def collect_rollout(self, buffer) -> dict:
        self._reset_episode_state()
        ep = super().collect_rollout(buffer)
        ep["tokens_emitted_cumulative"] = int(self._cum_tokens_emitted)
        ep["transfers_received_cumulative"] = int(self._cum_transfers_received)
        ratio = (
            self._cum_tokens_emitted / max(self._cum_transfers_received, 1)
        )
        ep["token_to_transfer_ratio"] = float(ratio)
        ep["per_city_tokens_emitted"] = self._per_city_emitted.tolist()
        ep["per_city_tokens_received"] = self._per_city_received.tolist()
        return ep

    def _extra_csv_columns(self) -> list[str]:
        cols = ["tokens_emitted_cum", "transfers_received_cum", "token_to_transfer_ratio"]
        cols += [f"tokens_emitted_city{i}" for i in range(self.n_agents)]
        cols += [f"tokens_received_city{i}" for i in range(self.n_agents)]
        return cols

    def _extra_csv_values(self, ep_metrics: dict) -> list:
        vals: list = [
            ep_metrics["tokens_emitted_cumulative"],
            ep_metrics["transfers_received_cumulative"],
            f"{ep_metrics['token_to_transfer_ratio']:.4f}",
        ]
        vals += list(ep_metrics["per_city_tokens_emitted"])
        vals += list(ep_metrics["per_city_tokens_received"])
        return vals

    def _maybe_shape_reward(
        self,
        r_raw: np.ndarray,
        infos: dict,
        agents: list[str],
        extras: dict,
    ) -> np.ndarray:
        """Apply token bonuses to senders."""
        r = r_raw.copy()
        token_fracs = extras.get("token_frac")
        if token_fracs is None:
            return r

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

            for sender_id, amount in received.items():
                share = amount / max(total_received, 1)
                allocated = int(round(tokens_to_emit * share))
                if allocated <= 0:
                    continue
                r[sender_id] += self.pi_cfg.token_exchange_rate * allocated
                self._cum_tokens_emitted += allocated
                self._per_city_emitted[i] += allocated
                self._per_city_received[sender_id] += allocated
            self._budget[i] -= tokens_to_emit
        return r
