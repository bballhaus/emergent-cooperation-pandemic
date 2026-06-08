"""Evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..env.pandemic_env import PandemicEnv


@dataclass
class EpisodeMetrics:
    welfare: float = 0.0
    total_deaths: float = 0.0
    total_unmet_vent_days: float = 0.0
    per_city_deaths: list[float] = field(default_factory=list)
    per_city_population: list[int] = field(default_factory=list)
    transfers_sent_total: int = 0
    gini_deaths_per_capita: float = 0.0
    worst_city_deaths_per_capita: float = 0.0

    def to_dict(self) -> dict:
        return {
            "welfare": self.welfare,
            "total_deaths": self.total_deaths,
            "total_unmet_vent_days": self.total_unmet_vent_days,
            "transfers_sent_total": self.transfers_sent_total,
            "gini_deaths_per_capita": self.gini_deaths_per_capita,
            "worst_city_deaths_per_capita": self.worst_city_deaths_per_capita,
            "per_city_deaths": list(self.per_city_deaths),
            "per_city_population": list(self.per_city_population),
        }


def gini(values: Iterable[float]) -> float:
    """Gini coefficient."""
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.where(arr < 0, 0.0, arr)
    if arr.sum() == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(sorted_arr)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def summarize_episode(env: PandemicEnv, unmet_weight: float = 0.01) -> EpisodeMetrics:
    """Summarize episode metrics from env cities."""
    cities = env.cities
    deaths = np.array([c.cumulative_deaths for c in cities], dtype=np.float64)
    unmet = np.array([c.cumulative_unmet_vent_days for c in cities], dtype=np.float64)
    pops = np.array([c.config.population for c in cities], dtype=np.float64)
    sent = sum(c.cumulative_sent for c in cities)
    per_capita = deaths / np.maximum(pops, 1.0)

    welfare = -float(deaths.sum() + unmet_weight * unmet.sum())
    return EpisodeMetrics(
        welfare=welfare,
        total_deaths=float(deaths.sum()),
        total_unmet_vent_days=float(unmet.sum()),
        per_city_deaths=deaths.tolist(),
        per_city_population=[int(p) for p in pops],
        transfers_sent_total=int(sent),
        gini_deaths_per_capita=gini(per_capita),
        worst_city_deaths_per_capita=float(per_capita.max()),
    )
