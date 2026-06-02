# Emergent Cooperation in Pandemic Resource Allocation

**CS 224R: Deep Reinforcement Learning — Stanford University, Spring 2026**  
Brooke Ballhaus · Riya Narain

---

## Overview

Can selfish reinforcement learning agents learn to share scarce medical resources during a pandemic? This project trains and evaluates multi-agent RL algorithms (IPPO, MAPPO, DQN, and a peer-incentive variant) on a multi-city SEIR-based pandemic simulator where each city controls its own ventilator, vaccine, and PPE stockpiles and must decide whether to transfer resources to cities in greater need.

The key finding: an **impact reward** — crediting transfers only up to the recipient's actual shortfall — combined with rich cross-city observations is sufficient to teach decentralized policy-gradient agents genuine cooperation, reducing deaths 4–11% relative to selfish baselines.

---

## Repository Structure

```
├── configs/                  # YAML environment configurations
│   ├── env_default.yaml      # 4-city, ventilators only (main experiments)
│   ├── env_multiresource.yaml
│   ├── env_multiresource_baseline.yaml
│   ├── env_multiresource_transfercost.yaml
│   ├── city_count_sweep.yaml
│   └── scarcity_sweep.yaml
├── src/
│   ├── env/                  # SEIR simulator, city model, config loader
│   ├── agents/               # IPPO, MAPPO, DQN, heuristic, peer-incentive
│   ├── train/                # PPO base, replay buffer, training loop
│   └── eval/                 # Metrics, learning curves, analysis
├── scripts/
│   ├── run_all_experiments.sh
│   ├── run_exchange_rate_sweep.py
│   ├── run_grid_parallel.py
│   ├── run_heuristic_baseline.py
│   ├── fetch_cdc_data.py     # Pulls calibrated city-level CDC data
│   └── fit_city_betas.py
├── runs/                     # Experimental results and visualizations
│   ├── aggregated.csv        # Per-algo, per-sweep summary (means + 95% CIs)
│   ├── aggregated_baseline.csv
│   ├── heuristic_baseline.json
│   ├── peer_token_flows.csv
│   └── *.png                 # Learning curves and token flow plots
├── tests/
├── docs/
│   └── experiment_log.md     # Full narrative of experiments E0–E7
└── requirements.txt
```

---

## Environment

Each city runs an independent SEIR model with calibrated contact rates and staggered 30-day demand surges (2× transmission multiplier, offset per city). Agents observe infection rates, hospitalization loads, and stockpiles for **all** cities, then decide weekly transfer quantities via a token-market mechanism.

| Parameter | Default |
|-----------|---------|
| Cities | 4 |
| Episode length | 180 days |
| Transit time | 1 day |
| Weekly supply per capita | 5×10⁻⁵ (tight scarcity) |
| Resources | Ventilators (+ vaccines, PPE in multi-resource) |

---

## Key Results (Default Config, 5 Seeds)

| Algorithm | Deaths (mean ± 95% CI) | Gini |
|-----------|------------------------|------|
| Heuristic (oracle) | 162,765 | 0.031 |
| **Peer** | **216,905 ± 5,215** | 0.074 |
| **IPPO** | **218,589 ± 8,723** | 0.049 |
| MAPPO | 261,754 ± 9,232 | 0.132 |
| DQN | 298,096 ± 21,237 | 0.090 |

Policy-gradient agents (IPPO, Peer) consistently outperform DQN. All RL agents remain below the heuristic oracle, which has access to global need information.

**Multi-resource (E5):** Adding vaccines and PPE yields 6–11% additional death reductions. Peer incentives achieve equity (Gini) approaching IPPO.

**Scope condition (E7):** A 30% transfer cost inverts the cooperation dilemma; RL agents collapse to hoarding and the impact reward becomes mis-specified.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Main dependencies: `torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `pyyaml`.

---

## Running Experiments

```bash
# Full default experiment suite
bash scripts/run_all_experiments.sh

# Heuristic baseline only
python scripts/run_heuristic_baseline.py

# City-count or scarcity sweeps in parallel
python scripts/run_grid_parallel.py --config configs/city_count_sweep.yaml

# Exchange-rate sweep (E6)
python scripts/run_exchange_rate_sweep.py
```

Results are written to `runs/` as CSV and JSON files with per-seed rollouts.

---

## Experiment Log

See [`docs/experiment_log.md`](docs/experiment_log.md) for a full narrative of experiments E0–E7, including failed approaches (E1, E2), the E3 breakthrough, and the transfer-cost failure mode (E7).

---

## Citation

If you build on this work:

```
@misc{ballhaus2026pandemic,
  title  = {Emergent Cooperation in Multi-Agent Pandemic Resource Allocation},
  author = {Ballhaus, Brooke and Narain, Riya},
  year   = {2026},
  note   = {CS 224R Final Project, Stanford University}
}
```
