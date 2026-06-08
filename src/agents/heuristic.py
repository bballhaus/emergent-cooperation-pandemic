"""Non-RL baselines.

ProportionalToNeedPolicy is the central-planner sanity floor referenced in the proposal:
each city covers its own currently-hospitalized patients first, then sends surplus
stockpile to other cities in proportion to their *observed* (public) infection burden
— since H of other cities is private, we proxy unmet need with normalized I.

SelfishHoardingPolicy keeps the entire stockpile local every step. This is the trivial
defection equilibrium; if IPPO collapses here, the SSD framing is biting.
"""

from __future__ import annotations

import numpy as np

from ..env.pandemic_env import PandemicEnv


class ProportionalToNeedPolicy:
    """Stateless heuristic. Operates directly on env state (white-box access)."""

    def __init__(self, env: PandemicEnv):
        self.env = env

    def act(self) -> dict[str, np.ndarray]:
        env = self.env
        n = env.n_cities
        actions: dict[str, np.ndarray] = {}
        infect_rates = np.array(
            [env.cities[j].state.I / max(env.cities[j].config.population, 1) for j in range(n)],
            dtype=np.float64,
        )

        for i, agent in enumerate(env.possible_agents):
            c = env.cities[i]
            stockpile = max(c.stockpile, 1)
            own_need = c.state.H
            own_share = float(min(own_need / stockpile, 1.0))
            remaining = max(1.0 - own_share, 0.0)

            other_rates = infect_rates.copy()
            other_rates[i] = 0.0
            total_other = other_rates.sum()
            if total_other > 1e-9 and remaining > 0:
                other_shares = remaining * other_rates / total_other
            else:
                other_shares = np.zeros(n)
                own_share = 1.0

            a = other_shares
            a[i] = own_share
            actions[agent] = a.astype(np.float32)
        return actions


class SelfishHoardingPolicy:
    """Allocate the entire stockpile locally every step. No transfers, ever."""

    def __init__(self, env: PandemicEnv):
        self.env = env

    def act(self) -> dict[str, np.ndarray]:
        n = self.env.n_cities
        actions: dict[str, np.ndarray] = {}
        for i, agent in enumerate(self.env.possible_agents):
            a = np.zeros(n, dtype=np.float32)
            a[i] = 1.0
            actions[agent] = a
        return actions
