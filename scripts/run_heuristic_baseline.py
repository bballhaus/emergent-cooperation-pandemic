"""Smoke test / sanity-floor experiment.

Runs the proportional-to-need heuristic and the selfish-hoarding heuristic on the
default 4-city env for one episode each, and prints welfare/equity/cooperation metrics.

This produces the milestone-report "first experiment" without any RL training: it shows
the env is wired up, the SSD framing is biting (selfish hoarding underperforms central
planning), and our metrics produce sensible numbers across both extremes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing as a package.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import json

from src.env import PandemicEnv, load_env_config
from src.agents import ProportionalToNeedPolicy, SelfishHoardingPolicy
from src.eval import summarize_episode


def run_episode(env: PandemicEnv, policy, seed: int = 0) -> dict:
    env.reset(seed=seed)
    while env.agents:
        actions = policy.act()
        env.step(actions)
    return summarize_episode(env, unmet_weight=env.config.unmet_reward_weight).to_dict()


def main() -> int:
    cfg = load_env_config("env_default.yaml")
    env = PandemicEnv(cfg)

    results = {}
    for name, policy_cls in [
        ("proportional_to_need", ProportionalToNeedPolicy),
        ("selfish_hoarding", SelfishHoardingPolicy),
    ]:
        seed_metrics = []
        for seed in range(5):
            policy = policy_cls(env)
            m = run_episode(env, policy, seed=seed)
            seed_metrics.append(m)
        # Simple cross-seed aggregation.
        results[name] = {
            "welfare_mean": sum(s["welfare"] for s in seed_metrics) / 5,
            "total_deaths_mean": sum(s["total_deaths"] for s in seed_metrics) / 5,
            "unmet_vent_days_mean": sum(s["total_unmet_vent_days"] for s in seed_metrics) / 5,
            "transfers_sent_mean": sum(s["transfers_sent_total"] for s in seed_metrics) / 5,
            "gini_mean": sum(s["gini_deaths_per_capita"] for s in seed_metrics) / 5,
            "worst_capita_mean": sum(s["worst_city_deaths_per_capita"] for s in seed_metrics) / 5,
            "per_seed": seed_metrics,
        }

    out_path = REPO / "runs" / "heuristic_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))

    print("\n=== Heuristic baselines (4 cities, 180 days, 5 seeds) ===")
    print(f"{'policy':<24} {'welfare':>12} {'deaths':>10} {'unmet':>12} {'transfers':>11} {'gini':>6} {'worst/cap':>10}")
    for name, r in results.items():
        print(
            f"{name:<24} "
            f"{r['welfare_mean']:>12.1f} "
            f"{r['total_deaths_mean']:>10.0f} "
            f"{r['unmet_vent_days_mean']:>12.0f} "
            f"{r['transfers_sent_mean']:>11.0f} "
            f"{r['gini_mean']:>6.3f} "
            f"{r['worst_capita_mean']:>10.5f}"
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
