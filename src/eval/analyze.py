"""Aggregate per-seed runs into a results table and plots."""

from __future__ import annotations

import argparse
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


DIR_RE = re.compile(
    r"^(?P<algo>[a-zA-Z_]+?)(?:_(?P<key>[a-zA-Z_]+)=(?P<val>[^_]+))?_seed(?P<seed>\d+)$"
)


def parse_run_dirs(root: Path):
    """Yield run metadata for matching subdirs."""
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
    """Mean and 95% CI per metric across seeds."""
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
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95"] = 1.96 * se
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["algo", "sweep", "n_seeds"])
    return pd.DataFrame(rows).sort_values(["sweep", "algo"]).reset_index(drop=True)


def peer_token_summary(root: Path, last_n: int = 10) -> pd.DataFrame:
    """Per-city token-flow summary for peer runs."""
    emit_re = re.compile(r"^tokens_emitted_city(\d+)$")
    recv_re = re.compile(r"^tokens_received_city(\d+)$")
    cells: dict = defaultdict(lambda: defaultdict(list))

    for algo, key, val, seed, csv in parse_run_dirs(root):
        if "peer" not in algo:
            continue
        df = pd.read_csv(csv)
        if df.empty or not any(emit_re.match(c) for c in df.columns):
            continue
        tail = df.tail(last_n)
        sweep_label = f"{key}={val}" if key else "default"
        for col in df.columns:
            m_e = emit_re.match(col)
            m_r = recv_re.match(col)
            if m_e:
                cells[(sweep_label, int(m_e.group(1)))]["emitted"].append(float(tail[col].mean()))
            elif m_r:
                cells[(sweep_label, int(m_r.group(1)))]["received"].append(float(tail[col].mean()))

    rows = []
    for (sweep_label, city), d in sorted(cells.items()):
        emitted = float(np.mean(d["emitted"])) if d["emitted"] else 0.0
        received = float(np.mean(d["received"])) if d["received"] else 0.0
        rows.append({
            "sweep": sweep_label,
            "city": city,
            "tokens_emitted_mean": emitted,
            "tokens_received_mean": received,
            "net_received_mean": received - emitted,
        })
    return pd.DataFrame(rows)


def plot_peer_tokens(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    sweeps = list(dict.fromkeys(df["sweep"]))
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5 * len(sweeps), 4), squeeze=False)
    for ax, sweep in zip(axes[0], sweeps):
        grp = df[df["sweep"] == sweep].sort_values("city")
        x = np.arange(len(grp))
        ax.bar(x - 0.2, grp["tokens_emitted_mean"], width=0.4, label="emitted (ack)")
        ax.bar(x + 0.2, grp["tokens_received_mean"], width=0.4, label="received (donate)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"city {c}" for c in grp["city"]])
        ax.set_title(f"peer token flow ({sweep})")
        ax.set_ylabel("tokens / episode (last-10 avg)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved peer token-flow plot to {out_path}")


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

    tokens = peer_token_summary(args.root, last_n=args.last_n)
    if not tokens.empty:
        tok_out = args.root / "peer_token_flows.csv"
        tokens.to_csv(tok_out, index=False)
        print(f"\n{tokens.to_string(index=False)}")
        print(f"Wrote {tok_out}")
        try:
            plot_peer_tokens(tokens, args.root / "peer_token_flows.png")
        except Exception as e:
            print(f"(skipped peer token plot: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
