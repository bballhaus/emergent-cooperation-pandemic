"""SEIR sanity checks: conservation, sign, ventilator effect."""

import numpy as np

from src.env.seir import SEIRParams, CompartmentState, step_seir


def test_population_conserved_over_time():
    state = CompartmentState.initial(1_000_000, initial_infected=100)
    params = SEIRParams()
    N0 = state.total()
    for _ in range(180):
        state, _ = step_seir(state, params, ventilators_used=0.0)
    assert abs(state.total() - N0) < 1.0, f"Population drift: {state.total()} vs {N0}"


def test_compartments_nonnegative():
    state = CompartmentState.initial(500_000, initial_infected=50)
    params = SEIRParams(beta=0.6)
    for _ in range(180):
        state, _ = step_seir(state, params, ventilators_used=0.0)
        for v in [state.S, state.E, state.I, state.H, state.R, state.D]:
            assert v >= 0, f"Negative compartment: {state}"


def test_ventilators_reduce_deaths():
    """Ventilators reduce deaths."""
    params = SEIRParams()
    deaths = {}
    for label, vent_supply in [("none", 0.0), ("full", 1e9)]:
        state = CompartmentState.initial(500_000, initial_infected=200)
        for _ in range(180):
            state, _ = step_seir(state, params, ventilators_used=vent_supply)
        deaths[label] = state.D
    assert deaths["full"] < deaths["none"], (
        f"Ventilators should reduce deaths but got full={deaths['full']:.1f} >= none={deaths['none']:.1f}"
    )


def test_zero_initial_infected_stays_zero():
    state = CompartmentState.initial(100_000, initial_infected=0)
    params = SEIRParams()
    for _ in range(30):
        state, _ = step_seir(state, params, ventilators_used=0.0)
    assert state.D < 1e-6 and state.H < 1e-6 and state.E < 1e-6, (
        f"Disease should not spread from zero infections, but got {state}"
    )
