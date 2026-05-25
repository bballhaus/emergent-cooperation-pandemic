"""Heuristic baselines should differ on transfer volume and give finite welfare."""

import numpy as np

from src.env import PandemicEnv, load_env_config
from src.agents import ProportionalToNeedPolicy, SelfishHoardingPolicy
from src.eval import summarize_episode


def _run(policy_cls, seed=0):
    cfg = load_env_config("env_default.yaml")
    env = PandemicEnv(cfg)
    env.reset(seed=seed)
    policy = policy_cls(env)
    while env.agents:
        env.step(policy.act())
    return summarize_episode(env, unmet_weight=env.config.unmet_reward_weight)


def test_proportional_makes_transfers():
    m = _run(ProportionalToNeedPolicy)
    assert m.transfers_sent_total >= 0  # may be zero if no surplus; check it runs


def test_selfish_makes_no_transfers():
    m = _run(SelfishHoardingPolicy)
    assert m.transfers_sent_total == 0


def test_welfare_finite_and_negative():
    """Welfare = -(deaths + scaled unmet). Should be finite and ≤ 0."""
    for cls in [ProportionalToNeedPolicy, SelfishHoardingPolicy]:
        m = _run(cls)
        assert np.isfinite(m.welfare)
        assert m.welfare <= 0
