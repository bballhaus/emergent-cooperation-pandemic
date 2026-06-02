"""Heuristic sanity-floor experiment, swept across scarcity and city count.

Runs the proportional-to-need and selfish-hoarding heuristics (no RL) and reports
welfare/equity/cooperation metrics. This is the *floor* the RL agents must beat, and the
central claim of the project is that the heuristic floor degrades at high scarcity — so we
need a baseline at *every sweep point*, not just the default config.

The sweep is controlled by env vars that mirror scripts/run_all_experiments.sh, so the
heuristic baselines line up with the RL run labels used by src/eval/analyze.py:

  bash:                                       sweep_label written to JSON
  (defaults)                                  -> "default"
  SCARCITIES="1.0e-5 5.0e-5 2.5e-4" ...        -> "weekly_supply_per_capita=<val>"
  N_CITIES="2 4 6 8" ...                       -> "n_cities=<val>"

Usage:
  python scripts/run_heuristic_baseline.py
  SCARCITIES="1.0e-5 2.5e-5 5.0e-5 1.0e-4 2.5e-4" python scripts/run_heuristic_baseline.py
  N_CITIES="2 4 6 8" python scripts/run_heuristic_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing as a package.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.env import PandemicEnv, build_env_config
from src.env.config_loader import load_yaml
from src.agents import ProportionalToNeedPolicy, SelfishHoardingPolicy
from src.eval import summarize_episode

BASE_SCARCITY = "5.0e-5"
BASE_NCITIES = "4"
N_SEEDS = int(os.environ.get("SEEDS", "5"))


def run_episode(env: PandemicEnv, policy, seed: int = 0) -> dict:
    env.reset(seed=seed)
    while env.agents:
        actions = policy.act()
        env.step(actions)
    return summarize_episode(env, unmet_weight=env.config.unmet_reward_weight).to_dict()


def sweep_label(scarcity: str, ncities: str) -> str:
    """Match the single key=val tag that run_all_experiments.sh / analyze.py use."""
    if ncities != BASE_NCITIES:
        return f"n_cities={ncities}"
    if scarcity != BASE_SCARCITY:
        return f"weekly_supply_per_capita={scarcity}"
    return "default"


def baselines_for(scarcity: str, ncities: str) -> dict:
    base_yaml = load_yaml(os.environ.get("BASE_CONFIG", "env_default.yaml"))
    base_yaml["weekly_supply_per_capita"] = float(scarcity)
    base_yaml["n_cities"] = int(ncities)
    env = PandemicEnv(build_env_config(base_yaml))

    results = {}
    for name, policy_cls in [
        ("proportional_to_need", ProportionalToNeedPolicy),
        ("selfish_hoarding", SelfishHoardingPolicy),
    ]:
        seed_metrics = []
        for seed in range(N_SEEDS):
            policy = policy_cls(env)
            seed_metrics.append(run_episode(env, policy, seed=seed))
        results[name] = {
            "welfare_mean": sum(s["welfare"] for s in seed_metrics) / N_SEEDS,
            "total_deaths_mean": sum(s["total_deaths"] for s in seed_metrics) / N_SEEDS,
            "unmet_vent_days_mean": sum(s["total_unmet_vent_days"] for s in seed_metrics) / N_SEEDS,
            "transfers_sent_mean": sum(s["transfers_sent_total"] for s in seed_metrics) / N_SEEDS,
            "gini_mean": sum(s["gini_deaths_per_capita"] for s in seed_metrics) / N_SEEDS,
            "worst_capita_mean": sum(s["worst_city_deaths_per_capita"] for s in seed_metrics) / N_SEEDS,
            "per_seed": seed_metrics,
        }
    return results


def main() -> int:
    scarcities = os.environ.get("SCARCITIES", BASE_SCARCITY).split()
    n_cities_list = os.environ.get("N_CITIES", BASE_NCITIES).split()

    all_results: dict[str, dict] = {}
    header = f"{'sweep':<34} {'policy':<22} {'welfare':>12} {'deaths':>10} {'unmet':>12} {'transfers':>11} {'gini':>6} {'worst/cap':>10}"
    print("\n=== Heuristic baselines (180 days, "
          f"{N_SEEDS} seeds) — floor the RL agents must beat ===")
    print(header)
    for scarcity in scarcities:
        for ncities in n_cities_list:
            label = sweep_label(scarcity, ncities)
            res = baselines_for(scarcity, ncities)
            all_results[label] = res
            for name, r in res.items():
                print(
                    f"{label:<34} {name:<22} "
                    f"{r['welfare_mean']:>12.1f} {r['total_deaths_mean']:>10.0f} "
                    f"{r['unmet_vent_days_mean']:>12.0f} {r['transfers_sent_mean']:>11.0f} "
                    f"{r['gini_mean']:>6.3f} {r['worst_capita_mean']:>10.5f}"
                )

    out_path = Path(os.environ.get("OUT_PATH", str(REPO / "runs" / "heuristic_baseline.json")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge into any existing baselines so the scarcity-pass and city-count-pass can be run
    # separately and accumulate into one file keyed by sweep label.
    merged: dict[str, dict] = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text())
        except ValueError:
            merged = {}
    merged.update(all_results)
    out_path.write_text(json.dumps(merged, indent=2, default=float))
    print(f"\nWrote {out_path} ({len(merged)} sweep points total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
