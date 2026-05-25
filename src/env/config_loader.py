"""Translate YAML config files in configs/ into an EnvConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .calibration import make_default_cities, COVID_DEFAULT
from .pandemic_env import EnvConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    with p.open() as f:
        return yaml.safe_load(f)


def build_env_config(yaml_dict: dict[str, Any]) -> EnvConfig:
    n = int(yaml_dict["n_cities"])
    cities = make_default_cities(
        n=n,
        base_params=COVID_DEFAULT,
        stagger_shocks=bool(yaml_dict.get("stagger_shocks", True)),
        shock_duration=int(yaml_dict.get("shock_duration", 30)),
        shock_magnitude=float(yaml_dict.get("shock_magnitude", 2.0)),
        initial_infected_per_city=int(yaml_dict.get("initial_infected_per_city", 50)),
        initial_stockpile=int(yaml_dict.get("initial_stockpile", 0)),
        max_days=int(yaml_dict.get("max_days", 180)),
    )
    return EnvConfig(
        cities=cities,
        max_days=int(yaml_dict.get("max_days", 180)),
        transit_days=int(yaml_dict.get("transit_days", 1)),
        weekly_supply_per_capita=float(yaml_dict.get("weekly_supply_per_capita", 5e-5)),
        death_reward_weight=float(yaml_dict.get("death_reward_weight", 1.0)),
        unmet_reward_weight=float(yaml_dict.get("unmet_reward_weight", 0.01)),
        reward_scale=float(yaml_dict.get("reward_scale", 1e-3)),
    )


def load_env_config(name: str = "env_default.yaml") -> EnvConfig:
    return build_env_config(load_yaml(name))
