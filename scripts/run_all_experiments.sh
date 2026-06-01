#!/usr/bin/env bash
# Run the 3-condition x 5-seed grid, optionally sweeping scarcity OR city count.
# Adjust ITERATIONS for compute budget. 1500 iters x 180 days = 270k env steps per agent.
#
# Run ONE sweep dimension at a time (the run-dir tag, and analyze.py's parser, hold a single
# key=val). Defaults give the baseline 4-city run at baseline scarcity.
#
# Usage:
#   bash scripts/run_all_experiments.sh                                  # baseline, 5 seeds
#   ITERATIONS=1500 bash scripts/run_all_experiments.sh
#   SCARCITIES="1.0e-5 2.5e-5 5.0e-5 1.0e-4 2.5e-4" bash scripts/run_all_experiments.sh   # scarcity sweep
#   N_CITIES="2 4 6 8" bash scripts/run_all_experiments.sh                                 # city-count sweep
#   ALGOS="ippo mappo peer" bash scripts/run_all_experiments.sh                            # drop DQN

set -euo pipefail

ITERATIONS=${ITERATIONS:-1500}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SCARCITIES=${SCARCITIES:-"5.0e-5"}            # baseline only; set multi-value for scarcity sweep
N_CITIES=${N_CITIES:-"4"}                     # baseline only; set multi-value for city-count sweep
ALGOS=${ALGOS:-"ippo mappo peer dqn"}         # DQN is the off-family (value-based) baseline
PYTHON=${PYTHON:-python}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

BASE_SCARCITY="5.0e-5"
BASE_NCITIES="4"

for scarcity in $SCARCITIES; do
  for ncities in $N_CITIES; do
    # Patch the env config on the fly with this scarcity + city count (temp YAML).
    tmp_config="/tmp/cs224r_env_s${scarcity}_n${ncities}.yaml"
    $PYTHON -c "
import yaml
cfg = yaml.safe_load(open('configs/env_default.yaml'))
cfg['weekly_supply_per_capita'] = float('${scarcity}')
cfg['n_cities'] = int('${ncities}')
yaml.safe_dump(cfg, open('${tmp_config}', 'w'))
"
    # Tag carries whichever dimension is non-default (single key=val so analyze.py can parse it).
    sweep_suffix=""
    if [ "$ncities" != "$BASE_NCITIES" ]; then
      sweep_suffix="_n_cities=${ncities}"
    elif [ "$scarcity" != "$BASE_SCARCITY" ]; then
      sweep_suffix="_weekly_supply_per_capita=${scarcity}"
    fi

    for algo in $ALGOS; do
      for seed in $SEEDS; do
        logdir="runs/${algo}${sweep_suffix}_seed${seed}"
        echo "=== $logdir ==="
        $PYTHON -m src.train.train \
          --algo "$algo" \
          --env-config "$tmp_config" \
          --seed "$seed" \
          --iterations "$ITERATIONS" \
          --log-dir "$logdir"
      done
    done
  done
done

echo ""
echo "Aggregating..."
$PYTHON -m src.eval.analyze runs/
