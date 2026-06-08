"""Rollout buffer for on-policy MARL training."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        rollout_len: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        has_token: bool = False,
        device: str = "cpu",
    ):
        self.T = rollout_len
        self.N = n_agents
        self.device = device
        self.has_token = has_token

        self.obs = np.zeros((self.T, self.N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.T, self.N, action_dim), dtype=np.float32)
        self.logp_alloc = np.zeros((self.T, self.N), dtype=np.float32)
        self.values = np.zeros((self.T, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.N), dtype=np.float32)
        self.dones = np.zeros((self.T, self.N), dtype=np.float32)
        self.global_state = np.zeros((self.T, global_state_dim), dtype=np.float32)
        if has_token:
            self.token_fracs = np.zeros((self.T, self.N), dtype=np.float32)
            self.logp_token = np.zeros((self.T, self.N), dtype=np.float32)

        self.advantages = np.zeros((self.T, self.N), dtype=np.float32)
        self.returns = np.zeros((self.T, self.N), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logp_alloc: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        global_state: np.ndarray,
        token_fracs: np.ndarray | None = None,
        logp_token: np.ndarray | None = None,
    ) -> None:
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.logp_alloc[t] = logp_alloc
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.global_state[t] = global_state
        if self.has_token:
            self.token_fracs[t] = token_fracs
            self.logp_token[t] = logp_token
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Compute GAE advantages and returns."""
        adv = np.zeros_like(self.rewards)
        gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            next_value = self.values[t + 1] if t + 1 < self.T else last_values
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values

    def iterate_minibatches(self, n_minibatches: int):
        """Yield random flattened minibatches."""
        TN = self.T * self.N
        idx = np.random.permutation(TN)
        mb_size = TN // n_minibatches

        flat_obs = self.obs.reshape(TN, -1)
        flat_actions = self.actions.reshape(TN, -1)
        flat_agent_ids = np.tile(np.arange(self.N), self.T)
        flat_logp_alloc = self.logp_alloc.reshape(TN)
        flat_values = self.values.reshape(TN)
        flat_advantages = self.advantages.reshape(TN)
        flat_returns = self.returns.reshape(TN)
        flat_global = np.repeat(self.global_state, self.N, axis=0)
        flat_token_fracs = self.token_fracs.reshape(TN) if self.has_token else None
        flat_logp_token = self.logp_token.reshape(TN) if self.has_token else None

        for k in range(n_minibatches):
            mb = idx[k * mb_size:(k + 1) * mb_size]
            batch = {
                "obs": torch.from_numpy(flat_obs[mb]).to(self.device),
                "actions": torch.from_numpy(flat_actions[mb]).to(self.device),
                "agent_ids": torch.from_numpy(flat_agent_ids[mb]).long().to(self.device),
                "logp_alloc_old": torch.from_numpy(flat_logp_alloc[mb]).to(self.device),
                "values_old": torch.from_numpy(flat_values[mb]).to(self.device),
                "advantages": torch.from_numpy(flat_advantages[mb]).to(self.device),
                "returns": torch.from_numpy(flat_returns[mb]).to(self.device),
                "global_state": torch.from_numpy(flat_global[mb]).to(self.device),
            }
            if self.has_token:
                batch["token_fracs"] = torch.from_numpy(flat_token_fracs[mb]).to(self.device)
                batch["logp_token_old"] = torch.from_numpy(flat_logp_token[mb]).to(self.device)
            yield batch

    def reset(self) -> None:
        self.ptr = 0
