"""PandemicEnv sanity checks: shape, simplex normalization, transfer accounting,
replenishment timing, episode termination."""

import numpy as np
import pytest

from src.env import PandemicEnv, load_env_config


def _make_env(n_cities=4):
    cfg = load_env_config("env_default.yaml")
    from src.env.config_loader import build_env_config, load_yaml
    yd = load_yaml("env_default.yaml")
    yd["n_cities"] = n_cities
    return PandemicEnv(build_env_config(yd))


def test_reset_returns_correct_shapes():
    env = _make_env(4)
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    for a, o in obs.items():
        assert o.shape == (env._obs_dim,)
    assert env.global_state().shape == (env.global_state_dim(),)


def test_step_with_uniform_actions_runs():
    env = _make_env(4)
    env.reset(seed=0)
    actions = {a: np.ones(env.n_cities, dtype=np.float32) / env.n_cities for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert len(rewards) == env.n_cities
    assert all(np.isfinite(r) for r in rewards.values())


def test_action_normalization_is_robust_to_unnormalized_input():
    env = _make_env(4)
    env.reset(seed=0)
    actions = {a: np.array([2.0, 0.0, 1.0, 1.0], dtype=np.float32) for a in env.agents}
    obs, rewards, _, _, _ = env.step(actions)
    assert all(np.isfinite(r) for r in rewards.values())


def test_episode_terminates_at_max_days():
    env = _make_env(4)
    env.reset(seed=0)
    steps = 0
    while env.agents:
        actions = {a: np.ones(env.n_cities, dtype=np.float32) / env.n_cities for a in env.agents}
        env.step(actions)
        steps += 1
        if steps > env.config.max_days + 5:
            pytest.fail("Env did not terminate at max_days")
    assert steps == env.config.max_days


def test_replenishment_happens_weekly():
    env = _make_env(4)
    env.reset(seed=0)
    stockpiles = []
    for _ in range(15):
        actions = {a: np.zeros(env.n_cities, dtype=np.float32) for a in env.agents}
        for a in actions:
            i = env.possible_agents.index(a)
            actions[a][i] = 1.0
        env.step(actions)
        stockpiles.append(sum(c.stockpile for c in env.cities))
    assert stockpiles[6] > stockpiles[5] or stockpiles[7] > stockpiles[6], (
        f"Expected stockpile to grow on weekly replenishment, got {stockpiles}"
    )


def test_transfer_arrives_after_transit_delay():
    env = _make_env(2)
    env.reset(seed=0)
    env.cities[0].stockpile = 100
    env.cities[1].stockpile = 0

    a0 = np.array([0.0, 1.0], dtype=np.float32)
    a1 = np.array([0.0, 1.0], dtype=np.float32)
    a1[1] = 1.0; a1[0] = 0.0
    actions = {"city_0": a0, "city_1": a1}
    env.step(actions)
    actions = {"city_0": np.array([1.0, 0.0], dtype=np.float32),
               "city_1": np.array([0.0, 1.0], dtype=np.float32)}
    env.step(actions)
    assert env.cities[1].stockpile >= 90, (
        f"Expected city_1 to have received transfer (~100), got {env.cities[1].stockpile}"
    )
    assert env.cities[0].cumulative_sent >= 90
