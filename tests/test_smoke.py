"""End-to-end smoke test: build env from the main config, step it, and train briefly.

CPU only, no network, no committed results; all outputs go to pytest's tmp_path.
"""

import numpy as np

from src.env import PandemicEnv, load_env_config
from src.agents import IPPOTrainer, PPOConfig, ProportionalToNeedPolicy


def test_env_default_reset_and_step_random_and_heuristic():
    env = PandemicEnv(load_env_config("env_default.yaml"))
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)

    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = {a: rng.dirichlet(np.ones(env.n_cities)).astype(np.float32) for a in env.agents}
        obs, rewards, terms, truncs, _ = env.step(actions)
        assert all(np.isfinite(r) for r in rewards.values())
        assert all(np.all(np.isfinite(o)) for o in obs.values())

    policy = ProportionalToNeedPolicy(env)
    for _ in range(5):
        _, rewards, _, _, _ = env.step(policy.act())
        assert all(np.isfinite(r) for r in rewards.values())


def test_tiny_ippo_training_writes_to_tmp_path(tmp_path):
    env = PandemicEnv(load_env_config("env_default.yaml"))
    log_dir = tmp_path / "ippo_smoke"
    cfg = PPOConfig(total_iterations=2, n_epochs=1, seed=0, log_dir=str(log_dir), device="cpu")
    IPPOTrainer(env, cfg).train()

    lines = (log_dir / "metrics.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + cfg.total_iterations  # header + one row per iteration
    assert (log_dir / "checkpoint.pt").exists()
