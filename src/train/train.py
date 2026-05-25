"""Single training-run entry point.

Usage:
  python -m src.train.train --algo ippo --env-config env_default.yaml --seed 0 \
        --iterations 200 --log-dir runs/ippo_seed0

  python -m src.train.train --algo mappo  ...
  python -m src.train.train --algo peer   ...
  python -m src.train.train --algo dqn    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.env import PandemicEnv, load_env_config
from src.agents import (
    IPPOTrainer, MAPPOTrainer,
    PeerIncentiveTrainer, PeerIncentiveConfig,
    DQNTrainer, DQNConfig,
    PPOConfig,
)


def build_trainer(algo: str, env, args):
    if algo == "ippo":
        cfg = PPOConfig(
            total_iterations=args.iterations,
            seed=args.seed,
            lr=args.lr,
            log_dir=args.log_dir,
            device=args.device,
        )
        return IPPOTrainer(env, cfg)
    if algo == "mappo":
        cfg = PPOConfig(
            total_iterations=args.iterations,
            seed=args.seed,
            lr=args.lr,
            log_dir=args.log_dir,
            device=args.device,
        )
        return MAPPOTrainer(env, cfg)
    if algo == "peer":
        cfg = PeerIncentiveConfig(
            total_iterations=args.iterations,
            seed=args.seed,
            lr=args.lr,
            log_dir=args.log_dir,
            device=args.device,
            token_budget=args.token_budget,
            token_exchange_rate=args.token_exchange_rate,
        )
        return PeerIncentiveTrainer(env, cfg)
    if algo == "dqn":
        cfg = DQNConfig(
            total_episodes=args.iterations,
            seed=args.seed,
            lr=args.lr,
            log_dir=args.log_dir,
            device=args.device,
        )
        return DQNTrainer(env, cfg)
    raise ValueError(f"Unknown algo: {algo}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["ippo", "mappo", "peer", "dqn"])
    p.add_argument("--env-config", default="env_default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=200,
                   help="PPO iterations (rollouts) or DQN episodes")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--token-budget", type=int, default=100)
    p.add_argument("--token-exchange-rate", type=float, default=1e-4)
    args = p.parse_args()

    if args.log_dir is None:
        args.log_dir = f"runs/{args.algo}_seed{args.seed}"

    env_cfg = load_env_config(args.env_config)
    env = PandemicEnv(env_cfg)
    trainer = build_trainer(args.algo, env, args)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
