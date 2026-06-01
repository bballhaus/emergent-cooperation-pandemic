"""Common PPO update for IPPO and MAPPO. Subclasses override critic input and reward shaping.

Hyperparameters follow Yu et al. 2022 §A.3:
  - 5 PPO epochs, 1 minibatch (full-batch update)
  - clip ratio 0.2, GAE lambda 0.95, gamma 0.99
  - value loss clipping enabled, value normalization implemented via running mean/std
  - shared trunk with Tanh activations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agents.networks import Actor, Critic
from ..env.pandemic_env import PandemicEnv
from .buffer import RolloutBuffer


@dataclass
class PPOConfig:
    rollout_len: int = 180          # one full episode per rollout (default episode length)
    total_iterations: int = 200     # 200 * 180 steps = 36k env steps per agent
    n_epochs: int = 5
    n_minibatches: int = 4
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 0.5
    # Advantages are normalized to unit std, so the policy-loss gradient is ~O(1) regardless
    # of reward scale; a 0.01 entropy bonus on a Dirichlet (entropy ~ -2) then dominates and
    # the policy never leaves near-uniform (milestone diagnosis: "exploration bonus dominates").
    # Start lower and anneal to zero so late training is reward-driven.
    ent_coef: float = 0.002
    ent_coef_final: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # 3e-4 (the usual PPO default) overshoots here: training drifts into a degenerate
    # high-transfer mode and deaths/unmet blow up after ~500 iters. 5e-5 is stable and keeps
    # improving monotonically through 2000 iters in this env.
    lr: float = 5e-5
    max_grad_norm: float = 0.5
    hidden: int = 128
    local_init_bias: float = 6.0  # init Dirichlet prior toward keeping stockpile local; see Actor
    seed: int = 0
    log_every: int = 10
    device: str = "cpu"
    log_dir: str = "runs/default"


class RunningNorm:
    """Streaming mean/std for value normalization (Yu et al. recommend this for stability)."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (np.sqrt(self.var) + 1e-8) + self.mean


class PPOBase:
    """Base class. Subclasses pick critic input and reward transformation."""

    name = "ppo_base"
    has_token_head = False

    def __init__(self, env: PandemicEnv, config: PPOConfig):
        self.env = env
        self.config = config
        self.n_agents = env.n_cities
        self.obs_dim = env._obs_dim
        self.action_dim = env._action_dim
        self.device = config.device

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.actor = Actor(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            hidden=config.hidden,
            has_token_head=self.has_token_head,
            local_init_bias=config.local_init_bias,
        ).to(self.device)
        self._ent_coef_now = config.ent_coef
        self.critic = Critic(
            input_dim=self._critic_input_dim(),
            hidden=config.hidden,
        ).to(self.device)

        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.lr
        )
        self.value_norm = RunningNorm()

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv = (self.log_dir / "metrics.csv").open("w")
        header = "iteration,env_steps,welfare,deaths,unmet,transfers,gini,worst_capita,policy_loss,value_loss,entropy"
        extra_cols = self._extra_csv_columns()
        if extra_cols:
            header += "," + ",".join(extra_cols)
        self._csv.write(header + "\n")

    # --- subclass hooks ---
    def _extra_csv_columns(self) -> list[str]:
        """Extra metrics.csv columns (e.g. peer-incentive token stats). Base logs none."""
        return []

    def _extra_csv_values(self, ep_metrics: dict) -> list:
        """Values for `_extra_csv_columns`, in the same order. Base logs none."""
        return []

    def _critic_input_dim(self) -> int:
        raise NotImplementedError

    def _critic_input(self, obs: torch.Tensor, agent_ids: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _reward_transform(self, rewards: np.ndarray) -> np.ndarray:
        """Map raw per-agent rewards (shape (N,)) → per-agent training rewards (shape (N,))."""
        raise NotImplementedError

    # --- rollout ---
    def _act(self, obs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Sample actions for all agents from current policy. Returns (actions, logp, value, extras)."""
        obs = torch.from_numpy(obs_np).float().to(self.device)
        agent_ids = torch.arange(self.n_agents, device=self.device, dtype=torch.long)
        with torch.no_grad():
            out = self.actor(obs, agent_ids)
            alloc = out.allocation_dist.sample()
            logp_alloc = out.allocation_dist.log_prob(alloc)
            critic_in = self._critic_input(
                obs, agent_ids,
                torch.from_numpy(self.env.global_state()).float().to(self.device).unsqueeze(0).expand(self.n_agents, -1)
            )
            value = self.critic(critic_in)
        extras: dict = {}
        if self.has_token_head and out.token_dist is not None:
            token_frac = out.token_dist.sample()
            logp_token = out.token_dist.log_prob(token_frac)
            extras["token_frac"] = token_frac.cpu().numpy()
            extras["logp_token"] = logp_token.cpu().numpy()
        return alloc.cpu().numpy(), logp_alloc.cpu().numpy(), value.cpu().numpy(), extras

    def collect_rollout(self, buffer: RolloutBuffer) -> dict:
        """Run one episode and fill `buffer`. Returns episode-summary stats."""
        from ..eval.metrics import summarize_episode
        obs_dict, _ = self.env.reset(seed=self.config.seed + buffer.ptr)  # vary seed per rollout
        buffer.reset()
        for t in range(self.config.rollout_len):
            obs_np = np.stack([obs_dict[a] for a in self.env.agents])
            actions_np, logp_np, value_np, extras = self._act(obs_np)

            actions_dict = {a: actions_np[i] for i, a in enumerate(self.env.agents)}
            agents_before = list(self.env.agents)
            obs_dict, rewards_dict, terms, truncs, infos = self.env.step(actions_dict)
            r_raw = np.array([rewards_dict[a] for a in agents_before], dtype=np.float32)
            r_train = self._maybe_shape_reward(r_raw, infos, agents_before, extras)
            done = np.array([float(terms[a]) for a in agents_before], dtype=np.float32)

            buffer.add(
                obs=obs_np,
                actions=actions_np,
                logp_alloc=logp_np,
                values=value_np,
                rewards=r_train,
                dones=done,
                global_state=self.env.global_state(),
                token_fracs=extras.get("token_frac") if self.has_token_head else None,
                logp_token=extras.get("logp_token") if self.has_token_head else None,
            )
            if not self.env.agents:
                break

        # Bootstrap value at end of rollout.
        if self.env.agents:
            obs_np = np.stack([obs_dict[a] for a in self.env.agents])
            obs_t = torch.from_numpy(obs_np).float().to(self.device)
            agent_ids = torch.arange(self.n_agents, device=self.device, dtype=torch.long)
            with torch.no_grad():
                critic_in = self._critic_input(
                    obs_t, agent_ids,
                    torch.from_numpy(self.env.global_state()).float().to(self.device).unsqueeze(0).expand(self.n_agents, -1)
                )
                last_values = self.critic(critic_in).cpu().numpy()
        else:
            last_values = np.zeros(self.n_agents, dtype=np.float32)
        buffer.compute_gae(last_values, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)

        ep = summarize_episode(self.env, unmet_weight=self.env.config.unmet_reward_weight)
        return ep.to_dict()

    def _maybe_shape_reward(
        self,
        r_raw: np.ndarray,
        infos: dict,
        agents: list[str],
        extras: dict,
    ) -> np.ndarray:
        """Hook for peer-incentive trainer; base trainers just apply _reward_transform."""
        return self._reward_transform(r_raw)

    # --- update ---
    def update(self, buffer: RolloutBuffer) -> dict:
        cfg = self.config
        # Update value normalization with current returns.
        self.value_norm.update(buffer.returns.reshape(-1))
        # Normalize advantages.
        adv = buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantages = adv

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(cfg.n_epochs):
            for batch in buffer.iterate_minibatches(cfg.n_minibatches):
                out = self.actor(batch["obs"], batch["agent_ids"])
                new_logp = out.allocation_dist.log_prob(batch["actions"])
                ratio = torch.exp(new_logp - batch["logp_alloc_old"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = out.allocation_dist.entropy().mean()

                # Token head shares the same advantage signal as the allocation head; both
                # decisions are made jointly at each step. This is the simplest credit
                # assignment that works — alternatives (separate value head per action
                # component) didn't show enough gain in pilot runs to justify the complexity.
                if self.has_token_head and out.token_dist is not None and "token_fracs" in batch:
                    new_logp_tok = out.token_dist.log_prob(batch["token_fracs"].clamp(1e-6, 1 - 1e-6))
                    ratio_tok = torch.exp(new_logp_tok - batch["logp_token_old"])
                    surr1_tok = ratio_tok * batch["advantages"]
                    surr2_tok = torch.clamp(ratio_tok, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * batch["advantages"]
                    policy_loss = policy_loss - torch.min(surr1_tok, surr2_tok).mean()
                    entropy = entropy + out.token_dist.entropy().mean()

                critic_in = self._critic_input(batch["obs"], batch["agent_ids"], batch["global_state"])
                new_values = self.critic(critic_in)
                returns_norm = torch.from_numpy(
                    self.value_norm.normalize(batch["returns"].cpu().numpy())
                ).float().to(self.device)
                v_clipped = batch["values_old"] + torch.clamp(
                    new_values - batch["values_old"], -cfg.value_clip, cfg.value_clip
                )
                vl1 = (new_values - returns_norm) ** 2
                vl2 = (v_clipped - returns_norm) ** 2
                value_loss = torch.max(vl1, vl2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - self._ent_coef_now * entropy
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optim.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                n_batches += 1
        for k in stats:
            stats[k] /= max(n_batches, 1)
        return stats

    # --- main loop ---
    def train(self) -> None:
        cfg = self.config
        buffer = RolloutBuffer(
            rollout_len=cfg.rollout_len,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            global_state_dim=self.env.global_state_dim(),
            has_token=self.has_token_head,
            device=self.device,
        )
        t0 = time.time()
        for it in range(1, cfg.total_iterations + 1):
            # Linearly anneal the entropy bonus from ent_coef -> ent_coef_final over training.
            frac = (it - 1) / max(cfg.total_iterations - 1, 1)
            self._ent_coef_now = cfg.ent_coef + frac * (cfg.ent_coef_final - cfg.ent_coef)

            ep_metrics = self.collect_rollout(buffer)
            stats = self.update(buffer)
            env_steps = it * cfg.rollout_len
            row = (
                f"{it},{env_steps},"
                f"{ep_metrics['welfare']:.2f},{ep_metrics['total_deaths']:.0f},"
                f"{ep_metrics['total_unmet_vent_days']:.0f},{ep_metrics['transfers_sent_total']:.0f},"
                f"{ep_metrics['gini_deaths_per_capita']:.4f},{ep_metrics['worst_city_deaths_per_capita']:.5f},"
                f"{stats['policy_loss']:.4f},{stats['value_loss']:.4f},{stats['entropy']:.4f}"
            )
            extra = self._extra_csv_values(ep_metrics)
            if extra:
                row += "," + ",".join(str(v) for v in extra)
            self._csv.write(row + "\n")
            self._csv.flush()
            if it % cfg.log_every == 0 or it == 1:
                elapsed = time.time() - t0
                print(
                    f"[{self.name}] it {it:3d} | welfare {ep_metrics['welfare']:>10.1f} | "
                    f"deaths {ep_metrics['total_deaths']:>7.0f} | transfers {ep_metrics['transfers_sent_total']:>9.0f} | "
                    f"gini {ep_metrics['gini_deaths_per_capita']:.3f} | "
                    f"pol {stats['policy_loss']:+.3f} val {stats['value_loss']:.3f} ent {stats['entropy']:+.3f} | "
                    f"{elapsed:.1f}s"
                )
        self._csv.close()

        # Save final checkpoints.
        ckpt = self.log_dir / "checkpoint.pt"
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_norm": {"mean": self.value_norm.mean, "var": self.value_norm.var, "count": self.value_norm.count},
            "config": cfg.__dict__,
        }, ckpt)
        print(f"Saved checkpoint to {ckpt}")
