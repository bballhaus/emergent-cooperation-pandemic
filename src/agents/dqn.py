"""Independent DQN baseline.

Discretizes the per-agent action into K options ranging from "all local" to "mostly
spread". Lets us test whether the cooperation/defection findings from PPO are
policy-gradient-specific or hold across algorithm families, as called out in the proposal.

Action discretization (per agent i, n_cities = N):
  0: 100% local                                  (full hoarding)
  1:  50% local, 50% spread equally across others
  2:  25% local, 75% spread equally across others
  3:  10% local, 90% spread equally across others
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.pandemic_env import PandemicEnv
from ..eval.metrics import summarize_episode


N_DISCRETE_ACTIONS = 4
LOCAL_FRACTIONS = [1.0, 0.5, 0.25, 0.10]


@dataclass
class DQNConfig:
    total_episodes: int = 200
    buffer_capacity: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 100
    target_update_every: int = 500   # in env steps
    hidden: int = 128
    seed: int = 0
    log_every: int = 10
    device: str = "cpu"
    log_dir: str = "runs/dqn"


def discrete_to_simplex(action_id: int, n_agents: int, agent_idx: int) -> np.ndarray:
    local_frac = LOCAL_FRACTIONS[action_id]
    simplex = np.full(n_agents, (1.0 - local_frac) / max(n_agents - 1, 1), dtype=np.float32)
    simplex[agent_idx] = local_frac
    return simplex


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(agent_ids, num_classes=self.n_agents).float()
        return self.net(torch.cat([obs, one_hot], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, transition: tuple) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


class DQNTrainer:
    name = "dqn"

    def __init__(self, env: PandemicEnv, config: DQNConfig):
        self.env = env
        self.cfg = config
        self.n_agents = env.n_cities
        self.obs_dim = env._obs_dim
        self.device = config.device

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        self.q = QNetwork(self.obs_dim, self.n_agents, N_DISCRETE_ACTIONS, config.hidden).to(self.device)
        self.q_target = QNetwork(self.obs_dim, self.n_agents, N_DISCRETE_ACTIONS, config.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=config.lr)

        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.env_steps = 0
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv = (self.log_dir / "metrics.csv").open("w")
        self._csv.write("episode,env_steps,welfare,deaths,unmet,transfers,gini,worst_capita,td_loss,epsilon\n")

    def _epsilon(self, ep: int) -> float:
        frac = min(ep / max(self.cfg.epsilon_decay_episodes, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def _select_actions(self, obs_np: np.ndarray, epsilon: float) -> np.ndarray:
        action_ids = np.empty(self.n_agents, dtype=np.int64)
        if random.random() < epsilon:
            action_ids[:] = np.random.randint(0, N_DISCRETE_ACTIONS, size=self.n_agents)
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).float().to(self.device)
                ids_t = torch.arange(self.n_agents, device=self.device, dtype=torch.long)
                q = self.q(obs_t, ids_t)
                action_ids = q.argmax(dim=-1).cpu().numpy()
        return action_ids

    def _td_update(self) -> float | None:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        obs, ids, acts, rews, next_obs, dones = zip(*batch)
        obs_t = torch.from_numpy(np.array(obs)).float().to(self.device)
        ids_t = torch.from_numpy(np.array(ids)).long().to(self.device)
        acts_t = torch.from_numpy(np.array(acts)).long().to(self.device).unsqueeze(-1)
        rews_t = torch.from_numpy(np.array(rews)).float().to(self.device)
        next_t = torch.from_numpy(np.array(next_obs)).float().to(self.device)
        dones_t = torch.from_numpy(np.array(dones)).float().to(self.device)

        q_sa = self.q(obs_t, ids_t).gather(1, acts_t).squeeze(-1)
        with torch.no_grad():
            q_next = self.q_target(next_t, ids_t).max(dim=-1).values
            target = rews_t + self.cfg.gamma * (1.0 - dones_t) * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()
        return loss.item()

    def train(self) -> None:
        t0 = time.time()
        last_td = 0.0
        for ep in range(1, self.cfg.total_episodes + 1):
            obs_dict, _ = self.env.reset(seed=self.cfg.seed + ep)
            eps = self._epsilon(ep)
            while self.env.agents:
                obs_np = np.stack([obs_dict[a] for a in self.env.agents])
                action_ids = self._select_actions(obs_np, eps)
                actions_dict = {
                    a: discrete_to_simplex(int(action_ids[i]), self.n_agents, i)
                    for i, a in enumerate(self.env.agents)
                }
                agents_before = list(self.env.agents)
                next_obs_dict, rewards, terms, truncs, _ = self.env.step(actions_dict)
                done_flag = float(terms[agents_before[0]])
                if self.env.agents:
                    next_obs_np = np.stack([next_obs_dict[a] for a in self.env.agents])
                else:
                    next_obs_np = obs_np  # terminal — bootstrap zeroed by (1-done)
                for i, a in enumerate(agents_before):
                    self.buffer.push((
                        obs_np[i].copy(),
                        i,
                        int(action_ids[i]),
                        float(rewards[a]),
                        next_obs_np[i].copy() if self.env.agents else obs_np[i].copy(),
                        done_flag,
                    ))
                obs_dict = next_obs_dict
                self.env_steps += 1
                td = self._td_update()
                if td is not None:
                    last_td = td
                if self.env_steps % self.cfg.target_update_every == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

            ep_metrics = summarize_episode(self.env, unmet_weight=self.env.config.unmet_reward_weight).to_dict()
            self._csv.write(
                f"{ep},{self.env_steps},"
                f"{ep_metrics['welfare']:.2f},{ep_metrics['total_deaths']:.0f},"
                f"{ep_metrics['total_unmet_vent_days']:.0f},{ep_metrics['transfers_sent_total']:.0f},"
                f"{ep_metrics['gini_deaths_per_capita']:.4f},{ep_metrics['worst_city_deaths_per_capita']:.5f},"
                f"{last_td:.4f},{eps:.3f}\n"
            )
            self._csv.flush()
            if ep % self.cfg.log_every == 0 or ep == 1:
                print(
                    f"[dqn] ep {ep:3d} | welfare {ep_metrics['welfare']:>10.1f} | "
                    f"deaths {ep_metrics['total_deaths']:>7.0f} | transfers {ep_metrics['transfers_sent_total']:>9.0f} | "
                    f"eps {eps:.2f} | td {last_td:.3f} | {time.time()-t0:.1f}s"
                )
        self._csv.close()
        torch.save({"q": self.q.state_dict(), "config": self.cfg.__dict__},
                   self.log_dir / "checkpoint.pt")
