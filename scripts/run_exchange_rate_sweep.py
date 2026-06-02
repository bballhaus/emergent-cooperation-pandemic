"""Peer token exchange-rate sensitivity sweep.

The peer-incentive algo trades cooperation tokens at a fixed --token-exchange-rate (reward
per token). This sweeps that rate to test how sensitive peer's welfare/equity are to the
price of cooperation. The default rate (5e-3) is already covered by the main
runs_multiresource/peer_seed* runs, so we only run the OTHER rates here and fold the
default in at analysis time.

Runs peer on the multi-resource impact config (env_multiresource.yaml) at the base point,
parallel across cores, resumable (skips runs whose metrics.csv is already complete).

  caffeinate -i .venv/bin/python scripts/run_exchange_rate_sweep.py --workers 9
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ITERATIONS = 1500
SEEDS = [0, 1, 2]  # 3 seeds (trimmed for compute); default-rate point reuses 5 main seeds
RATES = ["1.0e-3", "2.5e-3", "1.0e-2", "2.5e-2"]  # excludes default 5.0e-3
CONFIG = "env_multiresource.yaml"
EXPECTED_ROWS = ITERATIONS + 1


def is_complete(log_dir: Path) -> bool:
    csv = log_dir / "metrics.csv"
    if not csv.exists():
        return False
    with csv.open() as f:
        return sum(1 for _ in f) >= EXPECTED_ROWS


def run_job(job: dict) -> tuple[str, str]:
    log_dir = job["log_dir"]
    if is_complete(log_dir):
        return (log_dir.name, "skip")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [
        sys.executable, "-m", "src.train.train",
        "--algo", "peer",
        "--env-config", CONFIG,
        "--seed", str(job["seed"]),
        "--iterations", str(ITERATIONS),
        "--token-exchange-rate", job["rate"],
        "--log-dir", str(log_dir),
    ]
    with (log_dir / "train.log").open("w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return (log_dir.name, "ok" if proc.returncode == 0 else f"FAIL({proc.returncode})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--runs-dir", default="runs_exchange_rate")
    args = p.parse_args()

    runs_dir = REPO / args.runs_dir
    jobs = []
    # Dir name uses the analyze.py "<algo>_<key>=<val>_seed<n>" convention so the standard
    # aggregator parses the sweep without changes (val must contain no underscore).
    for rate in RATES:
        for seed in SEEDS:
            log_dir = runs_dir / f"peer_token_exchange_rate={rate}_seed{seed}"
            jobs.append({"rate": rate, "seed": seed, "log_dir": log_dir})

    todo = [j for j in jobs if not is_complete(j["log_dir"])]
    print(f"[xrate] {len(jobs)} jobs, {len(jobs) - len(todo)} complete, "
          f"{len(todo)} to run on {args.workers} workers", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[xrate] ({done}/{len(jobs)}) {status:>10}  {name}", flush=True)

    print("[xrate] all runs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
