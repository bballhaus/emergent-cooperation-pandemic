#!/usr/bin/env bash

set -euo pipefail

ITERATIONS=${ITERATIONS:-1500}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SCARCITIES=${SCARCITIES:-"5.0e-5"}
N_CITIES=${N_CITIES:-"4"}
ALGOS=${ALGOS:-"ippo mappo peer dqn"}
PYTHON=${PYTHON:-python}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

BASE_SCARCITY="5.0e-5"
BASE_NCITIES="4"

for scarcity in $SCARCITIES; do
  for ncities in $N_CITIES; do
    tmp_config="/tmp/cs224r_env_s${scarcity}_n${ncities}.yaml"
    $PYTHON -c "
import yaml
cfg = yaml.safe_load(open('configs/env_default.yaml'))
cfg['weekly_supply_per_capita'] = float('${scarcity}')
cfg['n_cities'] = int('${ncities}')
yaml.safe_dump(cfg, open('${tmp_config}', 'w'))
"
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
