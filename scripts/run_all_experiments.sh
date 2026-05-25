#!/usr/bin/env bash
# Run the full 3-condition x 5-seed x scarcity-sweep grid.
# Adjust ITERATIONS for compute budget. 200 iters × 180 days = 36k env steps per agent;
# at ~0.1s per iter (single env, CPU), one full run is ~20s + PPO update time.
#
# Usage:
#   bash scripts/run_all_experiments.sh                 # default scarcity, 5 seeds
#   ITERATIONS=400 SEEDS="0 1 2 3 4" bash scripts/run_all_experiments.sh
#   SCARCITIES="1.0e-5 5.0e-5 2.5e-4" bash scripts/run_all_experiments.sh

set -euo pipefail

ITERATIONS=${ITERATIONS:-200}
SEEDS=${SEEDS:-"0 1 2 3 4"}
SCARCITIES=${SCARCITIES:-"5.0e-5"}             # baseline only; set to multi-value for sweep
ALGOS=${ALGOS:-"ippo mappo peer"}              # add 'dqn' to also run DQN baseline
PYTHON=${PYTHON:-python}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

for scarcity in $SCARCITIES; do
  # Patch the env config on the fly with this scarcity value (write a temp YAML).
  tmp_config="/tmp/cs224r_env_scarcity_${scarcity}.yaml"
  $PYTHON -c "
import yaml
cfg = yaml.safe_load(open('configs/env_default.yaml'))
cfg['weekly_supply_per_capita'] = float('${scarcity}')
yaml.safe_dump(cfg, open('${tmp_config}', 'w'))
"
  for algo in $ALGOS; do
    for seed in $SEEDS; do
      tag="${algo}"
      if [ "$scarcity" != "5.0e-5" ]; then
        tag="${algo}_weekly_supply_per_capita=${scarcity}"
      fi
      logdir="runs/${tag}_seed${seed}"
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

echo ""
echo "Aggregating..."
$PYTHON -m src.eval.analyze runs/
