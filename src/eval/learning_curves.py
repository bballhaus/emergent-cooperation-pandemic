"""Plot training curves from per-seed metrics.csv files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import parse_run_dirs

DEFAULT_METRICS = ["deaths", "welfare", "transfers", "entropy"]


def _collect(root: Path, sweep: str, x_col: str, metric: str):
    """Per-algo (x, mean, std) for one metric."""
    series: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for algo, key, val, seed, csv in parse_run_dirs(root):
        sweep_label = f"{key}={val}" if key else "default"
        if sweep_label != sweep:
            continue
        df = pd.read_csv(csv)
        if df.empty or metric not in df.columns or x_col not in df.columns:
            continue
        series[algo].append(df[[x_col, metric]])

    out: dict[str, tuple] = {}
    for algo, frames in series.items():
        n = min(len(f) for f in frames)
        if n == 0:
            continue
        stack = np.stack([f[metric].to_numpy()[:n] for f in frames], axis=0)
        x = frames[0][x_col].to_numpy()[:n]
        out[algo] = (x, stack.mean(axis=0), stack.std(axis=0))
    return out


def _panel(ax, root: Path, sweep: str, x_col: str, metric: str) -> None:
    data = _collect(root, sweep, x_col, metric)
    for algo in sorted(data):
        x, mean, std = data[algo]
        ax.plot(x, mean, label=algo)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend()


def plot_curves(root: Path, sweep: str, x_col: str, metrics: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cols = min(len(metrics), 2)
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, metric in zip(flat, metrics):
        _panel(ax, root, sweep, x_col, metric)
    for ax in flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Learning curves ({sweep}) — mean ± 1 std across seeds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved learning curves to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="Directory containing per-seed run subdirs (typically runs/).")
    p.add_argument("--sweep", default="default",
                   help="Sweep label to plot (e.g. 'default', 'weekly_supply_per_capita=1.0e-5', 'n_cities=6').")
    p.add_argument("--metric", default=None,
                   help="Single metric to plot. If omitted, plots deaths/welfare/transfers/entropy.")
    p.add_argument("--x", default="iteration", choices=["iteration", "env_steps"],
                   help="X axis column.")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = p.parse_args()

    metrics = [args.metric] if args.metric else DEFAULT_METRICS
    tag = args.metric if args.metric else "curves"
    safe_sweep = args.sweep.replace("=", "").replace(".", "")
    out = args.out or (args.root / f"learning_{tag}_{safe_sweep}.png")
    plot_curves(args.root, args.sweep, args.x, metrics, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
