# AGENTS.md

## Summary

Stanford CS 224R (Spring 2026) final project by Brooke Ballhaus and Riya Narain.
It trains multi-agent RL policies (IPPO, MAPPO, DQN, and a peer-incentive
variant) on a multi-city SEIR pandemic simulator. Each city agent decides how
much of its ventilator (and, in multi-resource configs, vaccine and PPE)
stockpile to transfer to other cities. Results are compared against a
heuristic baseline. The narrative of experiments E0-E7 is in
`docs/experiment_log.md`.

## Repo map

- `src/env/`: simulator. `seir.py` (SEIR dynamics), `city.py`,
  `pandemic_env.py` (`PandemicEnv`, `EnvConfig`), `calibration.py` (city
  betas), `config_loader.py` (YAML in `configs/` to `EnvConfig`).
- `src/agents/`: `ippo.py`, `mappo.py`, `dqn.py`, `peer_incentive.py`,
  `heuristic.py`, `networks.py`.
- `src/train/`: `train.py` (single-run entry point), `ppo_base.py`, `buffer.py`.
- `src/eval/`: `analyze.py` (aggregate seeds to CSV), `learning_curves.py`,
  `metrics.py`.
- `configs/`: environment YAMLs (`env_default.yaml` is the main config) and
  sweep YAMLs (`city_count_sweep.yaml`, `scarcity_sweep.yaml`).
- `scripts/`: experiment drivers and data scripts (see below).
- `tests/`: pytest tests for the env, SEIR model, heuristic, and an
  end-to-end smoke test.
- `runs*/`: committed experiment outputs (see "Results").
- `data/raw/`, `data/processed/`: gitignored except `.gitkeep`.

## Setup

README states Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the repo root; modules are invoked as `python -m src...`.

## Tests

Framework is pytest (`tests/conftest.py` adds the repo root to `sys.path`).

```bash
python -m pytest tests/
python -m pytest tests/test_seir.py   # one file
```

`tests/test_smoke.py` is an end-to-end smoke test (builds the env from
`env_default.yaml`, steps it with random and heuristic actions, and trains IPPO
for 2 iterations into pytest's `tmp_path`). The whole suite runs on CPU in
about 10 seconds with no network or committed results.

CI: `.github/workflows/tests.yml` runs `python -m pytest -q` on Python 3.11
(CPU torch) for pushes and pull requests to `main`.

## Running experiments

Single small run (CPU by default; `--iterations` defaults to 200):

```bash
python -m src.train.train --algo ippo --env-config env_default.yaml \
  --seed 0 --iterations 20 --log-dir /tmp/ippo_smoke
```

`--algo` is one of `ippo mappo peer dqn`. Other flags: `--lr`, `--ent-coef`,
`--ent-coef-final`, `--local-init-bias`, `--device`, `--token-budget`,
`--token-exchange-rate`. If `--log-dir` is omitted it writes to
`runs/<algo>_seed<seed>`, so pass an explicit scratch dir for smoke tests.

Aggregate and plot existing per-seed runs:

```bash
python -m src.eval.analyze <runs_dir> [--last-n 10] [--out file.csv]
python -m src.eval.learning_curves <runs_dir> [--sweep default] [--out file.png]
```

Full sweeps (expensive: 1500 iterations x 4 algos x 5 seeds and more):

- `bash scripts/run_all_experiments.sh` - sequential; env overrides
  `ITERATIONS`, `SEEDS`, `SCARCITIES`, `N_CITIES`, `ALGOS`, `PYTHON`. Writes to
  `runs/` and then rewrites `runs/aggregated.csv`.
- `python scripts/run_grid_parallel.py` - parallel, resumable grid. Flags:
  `--workers`, `--base-config`, `--runs-dir`, `--algos`, `--default-only`,
  `--skip-city-sweep`. By default it runs both the scarcity and city-count
  sweeps.
- `python scripts/run_heuristic_baseline.py` - writes
  `runs/heuristic_baseline.json` (override with `OUT_PATH`, `SEEDS` env vars).
- `python scripts/run_exchange_rate_sweep.py` - peer exchange-rate sweep on
  `env_multiresource.yaml`; flags `--workers`, `--runs-dir`
  (default `runs_exchange_rate`).

## Results

`runs/`, `runs_milestone/`, `runs_multiresource/`,
`runs_multiresource_baseline/`, `runs_multiresource_transfercost/` and
`runs_exchange_rate/` hold committed experiment outputs (aggregated CSVs,
heuristic JSON, plots) backing the reported results. Per-seed run directories
are gitignored. Do not regenerate, overwrite or delete these files casually;
the default output paths of the sweep scripts and `analyze` point into them.
Use a separate `--runs-dir` / `--log-dir` / `--out` for new work.

## Conventions

- Config-driven: environment parameters live in YAML under `configs/`;
  `load_yaml` resolves bare filenames relative to `configs/`. Add a new YAML
  rather than hardcoding parameters. Sweep scripts write temporary derived
  configs to `/tmp`.
- Seeds are explicit (`--seed`); headline results use seeds 0-4 and
  1500 iterations.
- Code style: `from __future__ import annotations`, type hints, short
  docstrings, `argparse` CLIs with `main() -> int`, `pathlib` paths anchored
  to the repo root. No formatter or linter config is present.

## Guardrails

- Ask before launching any full sweep or multi-seed training run; they take
  hours of CPU.
- Do not edit or overwrite committed files in `runs*/` or the results tables
  in README / `docs/experiment_log.md` without being asked.
- Data provenance: `scripts/fetch_cdc_data.py` downloads CDC case data from
  data.cdc.gov to `data/raw/cdc_cases_by_state.csv`;
  `scripts/fit_city_betas.py` fits `data/processed/city_betas.json`, which
  `src/env/calibration.py` loads if present (otherwise it falls back to
  hardcoded betas). Do not commit files in `data/`; ask before downloading.
