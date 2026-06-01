# Experiment Log

Running record of the effort to improve RL performance. For each experiment we record
**(a) why we ran it** (what the previous result implied), **(b) the exact config/code
delta from the previous state**, **(c) the results**, and **(d) the decision** that leads
into the next experiment. This makes the chain of reasoning auditable end-to-end.

## Motivating problem

The proportional-to-need **heuristic beats every RL variant at every sweep point** (see
`runs/aggregated_baseline.csv`). Diagnosis identified two causes:

1. **Observation blind spot** — agents saw only other cities' infection rate (`I/N`),
   never their hospitalized load or stockpile, so they couldn't see *net need*.
2. **No targeting credit** — reward was purely local (own deaths + own unmet), so
   "send a ventilator to the city that needed it" was never reinforced. This also explains
   why DQN was the best RL variant *by hoarding* (transfers had no upside).

**Baseline (E0):** the pushed grid, `runs/aggregated_baseline.csv` — plain obs, no transfer
shaping, heterogeneous beta, 1500 iters, 5 seeds.

## Experiment index

| #  | Change                                   | Default-config result          | Verdict        |
|----|------------------------------------------|--------------------------------|----------------|
| E0 | Baseline (plain obs, local reward only)  | RL loses to heuristic; DQN hoards | reference   |
| E1 | Targeting reward (∝ gross giving)        | monotonically worse as weight↑ | ❌ reverted    |
| E2 | Rich obs alone (others' H/N, stockpile)  | +2–4% deaths (all algos)       | ❌ off by default |
| E3 | **Impact reward (saturating) + rich obs**| IPPO −4.1%, peer −3.7% deaths; Gini ~halved | ✅ **kept** |
| E4 | Full grid re-run under E3 config         | _in progress_                  | _pending_      |

All experiments below use the **default 4-city config** unless stated. "Δ" is vs the
relevant control in the same table. Lower deaths / lower Gini = better.

---

## E1 — Need-targeted transfer reward  ❌ REVERTED (negative result)

**Why this experiment:** the diagnosis said transfers earn no credit, so the first idea
was to directly pay an agent for sending to a needy city.

**Config/code delta from E0:** added `targeting_reward_weight` (EnvConfig field + step `1b`
bonus in `src/env/pandemic_env.py`, threaded through `config_loader.py` + `env_default.yaml`).
A sender earns `weight x (ventilators delivered) x (recipient I/N at arrival)`.

**A/B (peer, seed 0, 600 iters), sweeping the weight:**

| targeting weight | deaths | Gini  |
|------------------|-------:|------:|
| 0.0 (off)        | 232,040| 0.095 |
| 0.25             | 238,305| 0.127 |
| 0.5              | 236,129| 0.115 |
| 1.0              | 251,026| 0.114 |
| 5.0              | 280,230| 0.144 |

**Result:** monotonically *harmful* — even the smallest positive weight raises deaths and
worsens equity. Cause: the bonus rewards *gross* giving to any infected city, regardless of
whether the recipient lacked ventilators or the sender could spare them. PPO already
over-transfers, so paying it to transfer more amplifies the failure.

**Decision → next:** set `targeting_reward_weight: 0.0` (kept as a negative-result knob).
The failure says the *shape* is wrong (volume, not impact) — but maybe the agent also can't
*see* who needs help. So next we test observations in isolation (E2).

---

## E2 — Richer observations alone (others' H/N + stockpile/N)

**Why this experiment:** isolate whether the observation blind spot is the bottleneck,
independent of any reward shaping (E1 reverted, so reward is local-only here).

**Config/code delta from E1:** added `rich_observations` flag (`_obs_dim`, `_observe` in
`pandemic_env.py`). Each agent now sees, per other city, `H/N` (ventilator-demand proxy)
and `S_v/N` (supply) in addition to `I/N`, so it can compute *net need = demand − supply*.
Obs dim grows `9 + (n−1) → 9 + 3(n−1)`. Reward unchanged (local only).

**A/B (PPO family, 3 seeds, 1000 iters), mean over seeds:**

| algo  | obs   | deaths  | welfare  | Gini   | Δ deaths (rich−plain) |
|-------|-------|--------:|---------:|-------:|----------------------:|
| ippo  | plain | 235,037 | -256,072 | 0.1093 |                       |
| ippo  | rich  | 239,663 | -261,995 | 0.1272 | +4,627 (+2.0%)        |
| mappo | plain | 241,419 | -264,242 | 0.1302 |                       |
| mappo | rich  | 251,427 | -277,049 | 0.1180 | +10,008 (+4.1%)       |
| peer  | plain | 232,047 | -252,246 | 0.1044 |                       |
| peer  | rich  | 236,530 | -257,985 | 0.1126 | +4,484 (+1.9%)        |

**Result:** slightly *harmful* for every algo (+2–4% deaths). With a purely local reward
the agent gets no benefit from knowing others' need, so the extra `2(n−1)` dims act as
noise it overfits to. (A single-seed run showed lower Gini; it did not survive 3-seed
averaging.)

**Decision → next:** keep `rich_observations` off *by itself*. E1 and E2 together show the
two levers fail for the **same reason** — the reward gives no benefit to good transfers, so
(a) seeing need is useless and (b) rewarding volume backfires. Implication: we need a reward
that rewards transfer **impact**, and *then* the observations may finally be actionable.
That combined hypothesis is E3.

---

## E3 — Saturating impact reward (+ rich obs)  ✅ KEPT (works)

**Why this experiment:** direct consequence of E1+E2 — reward *useful* giving (capped at
real need so it can't be gamed by dumping), and pair it with the observations that expose
that need.

**Config/code delta from E2:** added `impact_reward_weight` (EnvConfig field + step `1c` in
`pandemic_env.py`). When a transfer arrives, the sender is credited
`weight x min(delivered, recipient_shortfall)`, `shortfall = H − stockpile_before`. Capped
at the recipient's actual shortfall, so the credit **saturates** — ventilators beyond need
earn nothing (fixing E1's dumping failure). Also flipped `rich_observations: true` so the
need signal is visible. Net default change from E0: `impact_reward_weight 0→1.0`,
`rich_observations false→true`, `targeting_reward_weight` stays 0.

**Weight sweep (peer, seed 0, 600 iters, rich obs):**

| impact weight | deaths  | Gini   | welfare  |
|---------------|--------:|-------:|---------:|
| 0.0           | 232,040 | 0.0949 | -252,237 |
| **1.0**       | 226,409 | 0.0557 | -245,032 |
| 2.0           | 232,466 | 0.0386 | -252,783 |
| 4.0           | 230,619 | 0.0743 | -250,420 |
| 8.0           | 230,711 | 0.0665 | -250,538 |

**A/B vs plain baseline (3 algos, 3 seeds, 1000 iters), mean over seeds:**

| algo  | variant          | deaths  | welfare  | Gini   | Δ deaths        | Δ Gini  |
|-------|------------------|--------:|---------:|-------:|----------------:|--------:|
| ippo  | plain, impact=0  | 235,037 | -256,072 | 0.1093 |                 |         |
| ippo  | rich, impact=1.0 | 225,318 | -243,636 | 0.0545 | **-9,718 (-4.1%)** | -0.055 |
| ippo  | rich, impact=2.0 | 228,704 | -247,968 | 0.0459 | -6,333 (-2.7%)  | -0.063  |
| mappo | plain, impact=0  | 241,419 | -264,242 | 0.1302 |                 |         |
| mappo | rich, impact=1.0 | 253,558 | -279,778 | 0.1082 | +12,138 (+5.0%) | -0.022  |
| mappo | rich, impact=2.0 | 249,529 | -274,621 | 0.1012 | +8,110 (+3.4%)  | -0.029  |
| peer  | plain, impact=0  | 232,047 | -252,246 | 0.1044 |                 |         |
| peer  | rich, impact=1.0 | 223,571 | -241,400 | 0.0641 | **-8,476 (-3.7%)** | -0.040 |
| peer  | rich, impact=2.0 | 230,967 | -250,866 | 0.0568 | -1,079 (-0.5%)  | -0.048  |

**Result:** at `w=1.0` the impact reward cuts **IPPO** deaths −4.1% and **peer** deaths
−3.7%, and roughly halves Gini for both — it teaches *decentralized* agents to cooperate,
closing toward MAPPO's centralized ceiling. **MAPPO** (already team-rewarded) does not
benefit (+5%): it already internalizes others' welfare, so the extra shaping is
redundant/destabilizing. RL still does not beat the heuristic, but the gap and the equity
disadvantage shrink markedly. `w=1.0` best for deaths, `w=2.0` best for Gini; chose `1.0`.

**Decision → next:** new default `impact_reward_weight: 1.0`, `rich_observations: true`.
Validate at scale: re-run the full grid for 5-seed/1500-iter headline numbers (E4).

---

## E4 — Full grid re-run under the E3 config  ⏳ IN PROGRESS

**Why this experiment:** E3's win is shown at the default config with 3 seeds / 1000 iters.
Confirm it holds across the full scarcity + city-count sweeps at publication settings, and
quantify the new RL-vs-heuristic gap everywhere.

**Config/code delta from E3:** none to the model — same `env_default.yaml` (impact=1.0,
rich obs). Operational: preserved the E0 numbers as `runs/aggregated_baseline.csv`, cleared
the 160 stale per-seed dirs (so the resumable runner re-trains rather than skipping), and
launched `scripts/run_grid_parallel.py --workers 8` (4 algos × 8 sweep points × 5 seeds ×
1500 iters).

**Results:** _pending — will record new `runs/aggregated.csv` vs `aggregated_baseline.csv`
deltas per algo/sweep, regenerate learning curves + token-flow plots, and summarize the
remaining heuristic gap here._

<!-- E4_RESULTS -->

---
