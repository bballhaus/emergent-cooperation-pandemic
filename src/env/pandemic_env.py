"""PettingZoo ParallelEnv for multi-city pandemic resource allocation.

Each city is an agent. Per daily timestep, an agent observes:
  - own SEIR compartments (normalized by population), own stockpile, own hospital capacity
  - for every other city, a public need/supply triple: infection rate (I/N),
    hospitalized load (H/N, the ventilator-demand proxy), and stockpile (S_v/N)
  - episode time signal

The agent's action is a non-negative vector of length `n_cities`. We normalize it to a
simplex; entry i (the agent's own index) is the fraction of its stockpile to use locally
this day, and entry j (j != i) is the fraction to transfer to city j (arriving after
`transit_days`).

Reward is per-city (selfish): a negative linear combination of new deaths and unmet
ventilator-days today. MAPPO/centralized algorithms can sum these into a team reward
in their own update loops; the env stays reward-structure-agnostic.

A central reserve replenishes the system every 7 days by distributing
`total_weekly_supply` ventilators proportional to population. Scarcity is controlled
by `weekly_supply_per_capita` in the config.

Demand shocks are staggered across cities (each city's beta is multiplied by
`shock_magnitude` during its `shock_duration`-day window), so unilateral hoarding is
locally tempting but globally costly — the structural SSD condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .city import City, CityConfig, IncomingTransfer
from .seir import SEIRParams, step_seir


@dataclass
class EnvConfig:
    cities: list[CityConfig] = field(default_factory=list)
    max_days: int = 180
    transit_days: int = 1                # delivery delay for inter-city transfers
    weekly_supply_per_capita: float = 5e-5  # central-reserve weekly inflow per resident
    death_reward_weight: float = 1.0     # cost of one death (in reward units)
    unmet_reward_weight: float = 0.01    # cost per unmet ventilator-day
    reward_scale: float = 1e-3           # global multiplier to keep PPO returns in a sane range
    targeting_reward_weight: float = 0.0 # bonus to a sender per (ventilator x recipient need)
                                         # when its transfer ARRIVES at a needy city. 0 = off
                                         # (selfish baseline); >0 shapes need-targeted giving.
    impact_reward_weight: float = 0.0    # bonus to a sender per delivered ventilator that
                                         # fell within the recipient's shortfall (saturating
                                         # impact credit). 0 = off; >0 rewards useful giving.
    rich_observations: bool = True       # True => each agent sees other cities' (I/N, H/N,
                                         # stockpile/N); False => only I/N (original obs).


class PandemicEnv(ParallelEnv):
    """PettingZoo ParallelEnv. Agents = cities, indexed 0..n-1, named 'city_i'."""

    metadata = {"name": "pandemic_multi_city_v0", "is_parallelizable": True}

    def __init__(self, config: EnvConfig):
        self.config = config
        self.n_cities = len(config.cities)
        if self.n_cities < 2:
            raise ValueError("Need at least 2 cities for a multi-agent env.")

        self.possible_agents = [f"city_{i}" for i in range(self.n_cities)]
        self.agents: list[str] = []
        self.cities: list[City] = [City.from_config(c) for c in config.cities]

        per_other = 3 if config.rich_observations else 1
        self._obs_dim = 9 + per_other * (self.n_cities - 1)  # see _observe()
        self._action_dim = self.n_cities

        self._obs_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

        self.day: int = 0
        self._rng = np.random.default_rng()
        # Per-step bookkeeping needed by peer-incentive wrapper / metrics.
        self.last_step_info: dict = {}

    # PettingZoo API ---------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.day = 0
        for c in self.cities:
            c.reset()
        self.agents = list(self.possible_agents)
        obs = {a: self._observe(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]):
        # 1. Deliver any transfers that arrive today (before allocation decisions use them).
        # Snapshot pre-arrival stockpiles so impact credit can measure how many arriving
        # ventilators actually fell within the recipient's unmet need.
        stockpile_before = [c.stockpile for c in self.cities]
        for c in self.cities:
            c.receive_arrivals(self.day)

        # 1b. Targeting credit (NEGATIVE-RESULT ABLATION, off by default): a sender earns a
        # bonus proportional to (ventilators delivered) x (recipient I/N). Rewards GROSS
        # giving, which amplified PPO's over-transfer and was harmful — kept only as a knob.
        targeting_bonus = np.zeros(self.n_cities, dtype=np.float64)
        if self.config.targeting_reward_weight > 0.0:
            for j, recipient in enumerate(self.cities):
                need_j = recipient.state.I / max(recipient.config.population, 1)
                for sender_id, amount in recipient.transfers_received_this_step.items():
                    if 0 <= sender_id < self.n_cities:
                        targeting_bonus[sender_id] += amount * need_j

        # 1c. Impact credit: a sender is rewarded only for the delivered ventilators that
        # fall within the recipient's actual shortfall (H - stockpile it already had).
        # min(delivered, shortfall) SATURATES once demand is met, so dumping surplus on a
        # city earns nothing — the key difference from the gross targeting bonus above.
        impact_bonus = np.zeros(self.n_cities, dtype=np.float64)
        if self.config.impact_reward_weight > 0.0:
            for j, recipient in enumerate(self.cities):
                received = recipient.transfers_received_this_step
                total_delivered = sum(received.values())
                if total_delivered <= 0:
                    continue
                shortfall = max(recipient.state.H - stockpile_before[j], 0.0)
                useful = min(float(total_delivered), shortfall)
                for sender_id, amount in received.items():
                    if 0 <= sender_id < self.n_cities:
                        impact_bonus[sender_id] += useful * (amount / total_delivered)

        # 2. Weekly central-reserve replenishment.
        if self.day > 0 and self.day % 7 == 0:
            self._replenish()

        # 3. Interpret each agent's action as a simplex over (use_locally, transfer_to_each_other).
        allocations: list[dict] = self._apply_actions(actions)

        # 4. Advance SEIR for each city using its locally-used ventilators.
        diagnostics_per_city: list[dict] = []
        rewards: dict[str, float] = {}
        for i, city in enumerate(self.cities):
            vents_local = allocations[i]["local"]
            new_state, diag = step_seir(
                city.state,
                city.config.seir_params,
                ventilators_used=vents_local,
                beta_multiplier=city.beta_multiplier(self.day),
            )
            new_deaths = new_state.D - city.state.D
            city.cumulative_deaths += new_deaths
            city.cumulative_unmet_vent_days += diag["unmet_vent_demand"]
            city.state = new_state
            diagnostics_per_city.append(diag)

            r = (
                -(
                    self.config.death_reward_weight * new_deaths
                    + self.config.unmet_reward_weight * diag["unmet_vent_demand"]
                )
                + self.config.targeting_reward_weight * targeting_bonus[i]
                + self.config.impact_reward_weight * impact_bonus[i]
            ) * self.config.reward_scale
            rewards[self.agents[i]] = float(r)

        # 5. Advance time, build outputs.
        self.day += 1
        terminated_flag = self.day >= self.config.max_days
        terminations = {a: terminated_flag for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = {a: self._observe(i) for i, a in enumerate(self.agents)}
        infos = {
            a: {
                "new_deaths": float(self.cities[i].state.D
                                    - (self.cities[i].cumulative_deaths
                                       - diagnostics_per_city[i]["deaths_today"])),
                "unmet_vent_demand": float(diagnostics_per_city[i]["unmet_vent_demand"]),
                "vent_coverage": float(diagnostics_per_city[i]["vent_coverage"]),
                "stockpile": int(self.cities[i].stockpile),
                "sent": int(allocations[i]["sent_total"]),
                "transfers_received": dict(self.cities[i].transfers_received_this_step),
            }
            for i, a in enumerate(self.agents)
        }
        self.last_step_info = infos

        if terminated_flag:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        pass  # text/plot renderer can be added if needed for debugging

    def close(self) -> None:
        pass

    # Helpers used by MAPPO critic and metrics --------------------------------
    def global_state(self) -> np.ndarray:
        """Concatenated city observations + day signal; centralized MAPPO critic input."""
        parts = [self._observe(i) for i in range(self.n_cities)]
        return np.concatenate(parts, dtype=np.float32)

    def global_state_dim(self) -> int:
        return self._obs_dim * self.n_cities

    # Internals ---------------------------------------------------------------
    def _observe(self, i: int) -> np.ndarray:
        c = self.cities[i]
        N = max(c.config.population, 1)
        own = np.array(
            [
                c.state.S / N,
                c.state.E / N,
                c.state.I / N,
                c.state.H / N,
                c.state.R / N,
                c.state.D / N,
                c.stockpile / N,
                c.config.hospital_capacity / N,
                self.day / max(self.config.max_days, 1),
            ],
            dtype=np.float32,
        )
        # Public need/supply signal for every other city. With rich_observations, expose
        # infection rate (I/N), hospitalized load (H/N, the ventilator-demand proxy), and
        # stockpile (S_v/N, their on-hand supply) — so an agent can compute net need
        # (demand minus supply), the signal the proportional-to-need heuristic exploits.
        # Otherwise expose only I/N (the original observation) for the ablation.
        others = []
        for j in range(self.n_cities):
            if j == i:
                continue
            Nj = max(self.cities[j].config.population, 1)
            if self.config.rich_observations:
                others.extend(
                    [
                        self.cities[j].state.I / Nj,
                        self.cities[j].state.H / Nj,
                        self.cities[j].stockpile / Nj,
                    ]
                )
            else:
                others.append(self.cities[j].state.I / Nj)
        others_arr = np.array(others, dtype=np.float32)
        return np.concatenate([own, others_arr], dtype=np.float32)

    def _apply_actions(self, actions: dict[str, np.ndarray]) -> list[dict]:
        """Convert each agent's action vector into integer allocations and execute transfers."""
        allocations: list[dict] = []
        for i, agent in enumerate(self.agents):
            a = np.asarray(actions[agent], dtype=np.float64).flatten()
            if a.shape[0] != self._action_dim:
                raise ValueError(f"Action for {agent} has shape {a.shape}, expected ({self._action_dim},)")
            a = np.clip(a, 0.0, None)
            s = a.sum()
            simplex = a / s if s > 1e-9 else np.ones_like(a) / len(a)

            stockpile = self.cities[i].stockpile
            # Integer floor on each slot, then push leftovers into the local-use bucket.
            raw = simplex * stockpile
            counts = np.floor(raw).astype(np.int64)
            counts[i] += stockpile - int(counts.sum())  # leftover to local

            local_use = int(counts[i])
            sent_total = 0
            for j in range(self.n_cities):
                if j == i or counts[j] <= 0:
                    continue
                sent = self.cities[i].send(
                    amount=int(counts[j]),
                    recipient=self.cities[j],
                    sender_id=i,
                    transit_days=self.config.transit_days,
                    day=self.day,
                )
                sent_total += sent
            # Stockpile was reduced by sends; remaining is the local-use cap.
            local_use = min(local_use, self.cities[i].stockpile)
            # Local use does not deplete stockpile (ventilators are reusable across days while in use),
            # but it does cap at current stockpile. Stockpile rolls forward.
            allocations.append({"local": local_use, "sent_total": sent_total})
        return allocations

    def _replenish(self) -> None:
        total_pop = sum(c.config.population for c in self.cities)
        total_weekly = int(round(total_pop * self.config.weekly_supply_per_capita * 7))
        if total_weekly <= 0:
            return
        # Proportional to population, with integer remainder going to the largest city.
        shares = np.array(
            [c.config.population / total_pop for c in self.cities], dtype=np.float64
        )
        raw = shares * total_weekly
        ints = np.floor(raw).astype(np.int64)
        ints[int(np.argmax(shares))] += total_weekly - int(ints.sum())
        for i, c in enumerate(self.cities):
            c.stockpile += int(ints[i])
