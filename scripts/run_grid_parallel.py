"""Parallel, resumable experiment grid runner.

Runs the scarcity sweep (5 levels x {ippo,mappo,peer,dqn} x 5 seeds) and the city-count
sweep (n=2,6,8 x 4 algos x 5 seeds) concurrently across CPU cores. Each run is pinned to a
single thread (OMP/MKL=1) so N parallel runs use N cores without oversubscription.

Resumable: a run whose metrics.csv already has the expected number of rows is skipped, so
relaunching after a crash/sleep only does the missing work. Each run's stdout goes to its
own <log_dir>/train.log (no interleaving).

Run under caffeinate so the machine won't idle-sleep mid-grid:
  caffeinate -i python scripts/run_grid_parallel.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

ITERATIONS = 1500
SEEDS = [0, 1, 2, 3, 4]
ALL_ALGOS = ["ippo", "mappo", "peer", "dqn"]
SCARCITIES = ["1.0e-5", "2.5e-5", "5.0e-5", "1.0e-4", "2.5e-4"]
N_CITIES = ["2", "6", "8"]
BASE_SCARCITY = "5.0e-5"
BASE_NCITIES = "4"
EXPECTED_ROWS = ITERATIONS + 1  # header + one row per iteration/episode

TMP = Path("/tmp/cs224r_grid_configs")


def sweep_suffix(scarcity: str, ncities: str) -> str:
    if ncities != BASE_NCITIES:
        return f"_n_cities={ncities}"
    if scarcity != BASE_SCARCITY:
        return f"_weekly_supply_per_capita={scarcity}"
    return ""


def make_config(scarcity: str, ncities: str, base_config: str) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((REPO / "configs" / base_config).read_text())
    cfg["weekly_supply_per_capita"] = float(scarcity)
    cfg["n_cities"] = int(ncities)
    base_tag = Path(base_config).stem
    out = TMP / f"{base_tag}_s{scarcity}_n{ncities}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def build_jobs(base_config: str, runs_dir: Path, algos: list[str],
               default_only: bool = False, skip_city_sweep: bool = False) -> list[dict]:
    """Enumerate (scarcity sweep at n=4) + (city sweep at base scarcity).

    default_only restricts to the single base point (4 cities, base scarcity) for a fast
    headline pass. skip_city_sweep keeps the scarcity sweep but drops the n_cities sweep
    (used for the baseline grid, whose city-count robustness is already covered by the
    impact grid).
    """
    if default_only:
        points: list[tuple[str, str]] = [(BASE_SCARCITY, BASE_NCITIES)]
    else:
        points = [(s, BASE_NCITIES) for s in SCARCITIES]
        if not skip_city_sweep:
            points += [(BASE_SCARCITY, n) for n in N_CITIES]
    jobs = []
    for scarcity, ncities in points:
        cfg = make_config(scarcity, ncities, base_config)
        suffix = sweep_suffix(scarcity, ncities)
        for algo in algos:
            for seed in SEEDS:
                log_dir = runs_dir / f"{algo}{suffix}_seed{seed}"
                jobs.append({"algo": algo, "seed": seed, "config": cfg, "log_dir": log_dir})
    return jobs


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
        "--algo", job["algo"],
        "--env-config", str(job["config"]),
        "--seed", str(job["seed"]),
        "--iterations", str(ITERATIONS),
        "--log-dir", str(log_dir),
    ]
    with (log_dir / "train.log").open("w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return (log_dir.name, "ok" if proc.returncode == 0 else f"FAIL({proc.returncode})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8, help="Concurrent runs (<= core count).")
    p.add_argument("--base-config", default="env_default.yaml",
                   help="Base env config in configs/ to sweep over.")
    p.add_argument("--runs-dir", default="runs",
                   help="Output directory (relative to repo) for per-run artifacts.")
    p.add_argument("--algos", default=",".join(ALL_ALGOS),
                   help="Comma-separated subset of algos to run.")
    p.add_argument("--default-only", action="store_true",
                   help="Run only the base point (4 cities, base scarcity) for a fast headline pass.")
    p.add_argument("--skip-city-sweep", action="store_true",
                   help="Keep the scarcity sweep but drop the n_cities sweep (baseline grid).")
    args = p.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    runs_dir = REPO / args.runs_dir
    jobs = build_jobs(args.base_config, runs_dir, algos, default_only=args.default_only,
                      skip_city_sweep=args.skip_city_sweep)
    todo = [j for j in jobs if not is_complete(j["log_dir"])]
    print(f"[grid] base={args.base_config} -> {runs_dir.name}/ | {len(jobs)} total jobs, "
          f"{len(jobs) - len(todo)} already complete, {len(todo)} to run on {args.workers} workers",
          flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[grid] ({done}/{len(jobs)}) {status:>10}  {name}", flush=True)

    print("[grid] all runs finished; aggregating...", flush=True)
    subprocess.run([sys.executable, "-m", "src.eval.analyze", str(runs_dir)], cwd=REPO)
    write_timings(runs_dir)
    return 0


def write_timings(runs_dir: Path) -> None:
    """Regenerate <runs_dir>/run_timings.csv from each run's train.log final timing line."""
    import re
    rows = []
    pat = re.compile(r"\|\s*([0-9.]+)s\s*$")
    for d in sorted(runs_dir.glob("*_seed*")):
        log = d / "train.log"
        if not log.exists():
            continue
        last = None
        for line in log.read_text().splitlines():
            m = pat.search(line)
            if m:
                last = m.group(1)
        if last:
            rows.append((d.name, last))
    out = runs_dir / "run_timings.csv"
    out.write_text("run,total_seconds\n" + "\n".join(f"{n},{t}" for n, t in rows) + "\n")
    print(f"[grid] wrote {out} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
