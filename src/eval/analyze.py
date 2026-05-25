"""Aggregate per-seed runs into a results table with 95% CIs and produce plots.

Expected layout in runs/:
  runs/<algo>_<sweep_key>=<value>_seed<seed>/metrics.csv
  runs/<algo>_seed<seed>/metrics.csv         # for default (no sweep)

Each CSV has columns including: welfare, deaths, transfers, gini, worst_capita.

Usage:
  python -m src.eval.analyze runs/                    # aggregate everything
  python -m src.eval.analyze runs/ --plot welfare     # also produce a welfare plot
"""

from __future__ import annotations

import argparse
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# Matches: <algo>_<key>=<val>_seed<n>  OR  <algo>_seed<n>
DIR_RE = re.compile(
    r"^(?P<algo>[a-zA-Z_]+?)(?:_(?P<key>[a-zA-Z_]+)=(?P<val>[^_]+))?_seed(?P<seed>\d+)$"
)


def parse_run_dirs(root: Path):
    """Yield (algo, sweep_key, sweep_val, seed, csv_path) for matching subdirs."""
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        csv = d / "metrics.csv"
        if not csv.exists():
            continue
        yield (
            m["algo"],
            m["key"],
            m["val"],
            int(m["seed"]),
            csv,
        )


def aggregate(root: Path, last_n: int = 10) -> pd.DataFrame:
    """For each (algo, sweep_key, sweep_val), compute mean + 95% CI of each metric over
    the last `last_n` iterations of each seed's run."""
    cells: dict = defaultdict(lambda: defaultdict(list))
    for algo, key, val, seed, csv in parse_run_dirs(root):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        tail = df.tail(last_n)
        sweep_label = f"{key}={val}" if key else "default"
        for col in ["welfare", "deaths", "unmet", "transfers", "gini", "worst_capita"]:
            if col in tail.columns:
                cells[(algo, sweep_label)][col].append(float(tail[col].mean()))

    rows = []
    for (algo, sweep_label), metric_dict in cells.items():
        row = {"algo": algo, "sweep": sweep_label, "n_seeds": len(next(iter(metric_dict.values())))}
        for col, values in metric_dict.items():
            arr = np.array(values, dtype=np.float64)
            mean = float(arr.mean())
            # 95% CI using t-dist would need scipy; use normal approx (1.96 SE) for n>=3.
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95"] = 1.96 * se
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["algo", "sweep", "n_seeds"])
    return pd.DataFrame(rows).sort_values(["sweep", "algo"]).reset_index(drop=True)


def plot_metric(root: Path, df: pd.DataFrame, metric: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, grp in df.groupby("algo"):
        ax.errorbar(
            grp["sweep"],
            grp[f"{metric}_mean"],
            yerr=grp[f"{metric}_ci95"],
            marker="o",
            label=algo,
            capsize=3,
        )
    ax.set_xlabel("sweep")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across runs (mean ± 95% CI of last 10 iters)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="Directory containing per-seed run subdirs (typically runs/).")
    p.add_argument("--last-n", type=int, default=10, help="Average over the last N iterations per seed.")
    p.add_argument("--plot", default=None, help="If set, also produce a plot of this metric (welfare/deaths/...).")
    p.add_argument("--out", type=Path, default=None, help="Output CSV (default: <root>/aggregated.csv).")
    args = p.parse_args()

    df = aggregate(args.root, last_n=args.last_n)
    out = args.out or (args.root / "aggregated.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")
    if args.plot:
        plot_metric(args.root, df, args.plot, args.root / f"plot_{args.plot}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
