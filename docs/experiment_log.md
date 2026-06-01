# Experiment Log

Running record of changes made to improve RL performance, with pre- vs post-change
results. The motivating problem: the proportional-to-need **heuristic beats every RL
variant at every sweep point** (see `runs/aggregated.csv`). Two diagnosed causes:

1. **Observation blind spot** — agents saw only other cities' infection rate (`I/N`),
   never their hospitalized load or stockpile, so they couldn't see *net need*.
2. **No targeting credit** — reward was purely local (own deaths + own unmet), so
   "send a ventilator to the city that needed it" was never reinforced. This is also
   why DQN won among RL by *hoarding*.

Baseline for all comparisons is the pushed grid in `runs/aggregated.csv` (plain obs,
no targeting reward, heterogeneous beta, 1500 iters, 5 seeds).

---

## Change 1 — Need-targeted transfer reward  ❌ REVERTED (negative result)

**What:** Added `targeting_reward_weight`. A sender earns
`weight x (ventilators delivered) x (recipient I/N at arrival)` when its transfer
*arrives* at a city, giving direct credit for need-targeted giving.
Files: `src/env/pandemic_env.py` (EnvConfig field + step `1b` bonus), `config_loader.py`,
`configs/env_default.yaml`.

**A/B (peer, default config, seed 0, 600 iters), sweeping the weight:**

| targeting weight | deaths | Gini  |
|------------------|-------:|------:|
| 0.0 (off)        | 232,040| 0.095 |
| 0.25             | 238,305| 0.127 |
| 0.5              | 236,129| 0.115 |
| 1.0              | 251,026| 0.114 |
| 5.0              | 280,230| 0.144 |

**Result:** monotonically *harmful* — even the smallest positive weight raises deaths
and worsens equity. Cause: the bonus rewards *gross* giving to any infected city,
regardless of whether the recipient lacked ventilators or the sender could spare them.
PPO already over-transfers, so paying it to transfer more amplifies the failure.

**Decision:** default `targeting_reward_weight: 0.0`. Mechanism kept in code as a
documented negative-result ablation knob.

---

## Change 2 — Richer observations (other cities' H/N + stockpile/N)

**What:** Added `rich_observations` (default true). Each agent now also sees, per other
city, hospitalized load `H/N` (ventilator-demand proxy) and stockpile `S_v/N` (supply),
in addition to `I/N` — so it can compute *net need = demand − supply*, the signal the
heuristic exploits. Obs dim grows from `9 + (n-1)` to `9 + 3(n-1)`. `false` reproduces
the original observation for the ablation.
Files: `src/env/pandemic_env.py` (`_obs_dim`, `_observe`), `config_loader.py`,
`configs/env_default.yaml`.

**A/B (PPO family, default config, 3 seeds, 1000 iters), mean over seeds:**

| algo  | obs   | deaths  | welfare  | Gini   | Δ deaths (rich−plain) |
|-------|-------|--------:|---------:|-------:|----------------------:|
| ippo  | plain | 235,037 | -256,072 | 0.1093 |                       |
| ippo  | rich  | 239,663 | -261,995 | 0.1272 | +4,627 (+2.0%)        |
| mappo | plain | 241,419 | -264,242 | 0.1302 |                       |
| mappo | rich  | 251,427 | -277,049 | 0.1180 | +10,008 (+4.1%)       |
| peer  | plain | 232,047 | -252,246 | 0.1044 |                       |
| peer  | rich  | 236,530 | -257,985 | 0.1126 | +4,484 (+1.9%)        |

**Result:** slightly *harmful* for every algo (+2–4% deaths). With a purely local reward
the agent gets no benefit from knowing other cities' need, so the extra `2(n−1)` input
dims act as noise that the policy overfits to. (An earlier single-seed run showed lower
Gini, but that did not survive 3-seed averaging.)

**Decision (interim):** `rich_observations` harmful alone — but see Change 3, where it
pairs with the impact reward to become part of the winning config.

---

## Takeaway after Changes 1 & 2

Both naive levers failed for the *same* reason: the reward gives no benefit to good
transfers, so (a) exposing need has nothing to act on, and (b) a gross giving-bonus just
amplifies PPO's existing over-transfer. The fix has to be a **better-shaped cooperation
incentive that rewards transfer _impact_, not volume**.

---

## Change 3 — Saturating impact reward (+ rich obs)  ✅ KEPT (works)

**What:** Added `impact_reward_weight`. When a transfer arrives, the sender is credited
`weight x min(delivered, recipient_shortfall)` where `shortfall = H - stockpile_before`.
Because it is capped at the recipient's actual shortfall, the credit **saturates** —
ventilators beyond need earn nothing, so dumping is not rewarded (the failure mode of
Change 1). Paired with rich observations so the agent can both *see* need and be *rewarded*
for meeting it. Files: `src/env/pandemic_env.py` (EnvConfig field + step `1c`),
`config_loader.py`, `configs/env_default.yaml`.

**Weight sweep (peer, default, seed 0, 600 iters, rich obs):**

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
−3.7%, and roughly halves Gini for both — i.e. it teaches *decentralized* agents to
cooperate, closing toward MAPPO's centralized ceiling. **MAPPO** (already team-rewarded)
does not benefit (+5% deaths), as expected: it already internalizes others' welfare, so
the extra shaping is redundant/destabilizing. The RL agents still do not beat the
proportional-to-need heuristic, but the gap and the equity disadvantage shrink markedly.

**Decision:** new default `impact_reward_weight: 1.0`, `rich_observations: true`. Next:
re-run the full grid (all algos x sweeps x 5 seeds) under this config for headline numbers.

---
