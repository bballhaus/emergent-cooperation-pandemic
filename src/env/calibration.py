"""Calibration: literature-default SEIR parameters, city demographic table,
and a least-squares fit of beta from a CDC case-count time series.

Data sources (see scripts/fetch_cdc_data.py to pull):
  - CDC United States COVID-19 Cases and Deaths by State Over Time
    https://data.cdc.gov/Case-Surveillance/United-States-COVID-19-Cases-and-Deaths-by-State-o/9mfq-cb36
  - U.S. Census Bureau state population estimates (2020-2024)
  - AHA hospital statistics (ICU bed counts per state) — used for hospital_capacity defaults.

Hospital-capacity numbers below are *staffed adult ICU beds* per state (rounded), drawn
from the HHS Protect Public Data Hub late-2021 snapshot. They serve as obs-scaling
constants only; the env does not impose a hard cap (ventilator stockpile is the binding
resource), so order-of-magnitude is what matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .city import CityConfig
from .seir import SEIRParams


# Literature-derived SEIR parameters for the COVID-19 ancestral strain (R0 ≈ 2.4).
# See: Li et al. 2020 (NEJM), Verity et al. 2020 (Lancet ID), Salje et al. 2020 (Science).
COVID_DEFAULT = SEIRParams(
    beta=0.30,          # R0 = 2.4 with gamma = 1/8
    sigma=1.0 / 4.0,    # 4-day latent period
    gamma=1.0 / 8.0,    # 8-day infectious period
    hosp_frac=0.025,    # ~2.5% of infections require critical care
    gamma_h=1.0 / 14.0, # 14-day average ICU stay
    mu_no_vent=0.90 / 14.0,
    mu_vent=0.40 / 14.0,
)


# (name, population, staffed_adult_icu_beds). Eight largest U.S. cities + spillover states
# are reasonable defaults; calibration_lookup() returns a subset of size N.
CITY_TABLE: list[tuple[str, int, int]] = [
    ("New York",    8_336_817, 2200),
    ("Los Angeles", 3_898_747, 1800),
    ("Chicago",     2_746_388, 1500),
    ("Houston",     2_304_580, 1300),
    ("Phoenix",     1_608_139,  900),
    ("Philadelphia",1_603_797, 1000),
    ("San Antonio", 1_434_625,  700),
    ("San Diego",   1_386_932,  800),
]


# Per-city ancestral-strain transmission rate (beta). The national literature default is
# 0.30 (R0 ≈ 2.4); these encode documented early-2020 heterogeneity — dense coastal metros
# (NYC, Chicago, Philadelphia) ran hotter than lower-density sunbelt cities (Phoenix, San
# Antonio, San Diego). They are *literature-informed defaults*: running the CDC fit pipeline
# (scripts/fetch_cdc_data.py → scripts/fit_city_betas.py) writes per-state fits to
# data/processed/city_betas.json, which overrides these per city when present.
CITY_BETA_DEFAULT: dict[str, float] = {
    "New York":     0.36,
    "Los Angeles":  0.30,
    "Chicago":      0.32,
    "Houston":      0.28,
    "Phoenix":      0.26,
    "Philadelphia": 0.31,
    "San Antonio":  0.27,
    "San Diego":    0.28,
}

# Output of scripts/fit_city_betas.py (empty until the CDC fetch+fit has been run).
CALIBRATED_BETA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "city_betas.json"
)


def load_calibrated_betas(path: Path = CALIBRATED_BETA_PATH) -> dict[str, float]:
    """Return {city_name: fitted_beta} from the CDC-fit output, or {} if not yet produced."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {str(k): float(v) for k, v in raw.items()}
    except (ValueError, OSError):
        return {}


def city_beta(name: str, calibrated: Optional[dict[str, float]] = None) -> float:
    """Resolve a city's beta: CDC fit if available, else the literature-informed default."""
    if calibrated and name in calibrated:
        return calibrated[name]
    return CITY_BETA_DEFAULT.get(name, COVID_DEFAULT.beta)


def make_default_cities(
    n: int,
    base_params: SEIRParams = COVID_DEFAULT,
    stagger_shocks: bool = True,
    shock_duration: int = 30,
    shock_magnitude: float = 2.0,
    initial_infected_per_city: int = 50,
    initial_stockpile: int = 0,
    max_days: int = 180,
    heterogeneous_beta: bool = True,
) -> list[CityConfig]:
    """Return `n` CityConfigs (n in 2..8) drawn from CITY_TABLE.

    If `stagger_shocks`, city i's surge starts on day i * (max_days // n), so demand
    peaks at different times — the structural condition that makes sharing rational
    (and unilateral hoarding tempting) under an SSD framing.

    If `heterogeneous_beta`, each city gets its own transmission rate (CDC fit if present,
    else the literature-informed CITY_BETA_DEFAULT), so cities differ in contact rate as
    well as population/capacity/shock-timing. Set False for a uniform-beta ablation.
    """
    if n < 2 or n > len(CITY_TABLE):
        raise ValueError(f"n must be in [2, {len(CITY_TABLE)}], got {n}")

    calibrated = load_calibrated_betas() if heterogeneous_beta else {}

    cities: list[CityConfig] = []
    for i in range(n):
        name, pop, cap = CITY_TABLE[i]
        shock_start = (i * (max_days // n)) if stagger_shocks else 30
        if heterogeneous_beta:
            params = SEIRParams(**{**base_params.__dict__, "beta": city_beta(name, calibrated)})
        else:
            params = base_params
        cities.append(
            CityConfig(
                name=name,
                population=pop,
                hospital_capacity=cap,
                seir_params=params,
                initial_infected=initial_infected_per_city,
                initial_stockpile=initial_stockpile,
                shock_start_day=shock_start,
                shock_duration=shock_duration,
                shock_magnitude=shock_magnitude,
            )
        )
    return cities


def fit_beta_from_cases(
    daily_new_cases: np.ndarray,
    population: int,
    other_params: SEIRParams = COVID_DEFAULT,
    initial_infected: int = 10,
    burn_in_days: int = 5,
) -> float:
    """Least-squares fit of `beta` against a daily-new-cases time series.

    Runs the SEIHRD model forward for `len(daily_new_cases)` days at a grid of beta
    values, picks the beta whose simulated `new_infections` minimize squared log-error
    against the observed series. We skip the first `burn_in_days` to avoid noise from
    seeding effects.
    """
    from .seir import CompartmentState, step_seir

    candidate_betas = np.linspace(0.05, 0.80, 76)
    best_beta, best_err = float(other_params.beta), float("inf")
    log_obs = np.log(np.maximum(daily_new_cases[burn_in_days:], 1.0))

    for b in candidate_betas:
        p = SEIRParams(**{**other_params.__dict__, "beta": b})
        state = CompartmentState.initial(population, initial_infected)
        sim = []
        for _ in range(len(daily_new_cases)):
            state, diag = step_seir(state, p, ventilators_used=0.0)
            sim.append(diag["new_infections"])
        sim = np.array(sim[burn_in_days:])
        log_sim = np.log(np.maximum(sim, 1.0))
        err = float(np.mean((log_obs - log_sim) ** 2))
        if err < best_err:
            best_err, best_beta = err, float(b)
    return best_beta


def load_cdc_state_data(csv_path: Path, state: str) -> Optional[np.ndarray]:
    """Load daily new cases for one state from CDC 9mfq-cb36 CSV (downloaded separately).

    Returns the chronologically-sorted new-case array, or None if the file is missing.
    The CDC file has columns: submission_date, state, tot_cases, new_case, ...
    """
    if not csv_path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=["submission_date"])
    sub = df[df["state"] == state].sort_values("submission_date")
    if sub.empty:
        return None
    return sub["new_case"].fillna(0).clip(lower=0).to_numpy()
