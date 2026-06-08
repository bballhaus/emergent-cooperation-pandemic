"""SEIHRD dynamics with ventilator-conditional mortality."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SEIRParams:
    beta: float = 0.30
    sigma: float = 1.0 / 4.0
    gamma: float = 1.0 / 8.0
    hosp_frac: float = 0.025
    gamma_h: float = 1.0 / 14.0
    mu_no_vent: float = 0.90 / 14.0
    mu_vent: float = 0.40 / 14.0
    vaccine_efficacy: float = 0.9
    ppe_max_reduction: float = 0.5


@dataclass
class CompartmentState:
    """Continuous-valued compartment counts."""
    S: float
    E: float
    I: float
    H: float
    R: float
    D: float

    def total(self) -> float:
        return self.S + self.E + self.I + self.H + self.R + self.D

    def to_array(self) -> np.ndarray:
        return np.array([self.S, self.E, self.I, self.H, self.R, self.D], dtype=np.float64)

    @classmethod
    def initial(cls, population: int, initial_infected: int = 10) -> "CompartmentState":
        return cls(
            S=float(population - initial_infected),
            E=0.0,
            I=float(initial_infected),
            H=0.0,
            R=0.0,
            D=0.0,
        )


def step_seir(
    state: CompartmentState,
    params: SEIRParams,
    ventilators_used: float,
    beta_multiplier: float = 1.0,
    vaccines_used: float = 0.0,
    ppe_used: float = 0.0,
) -> tuple[CompartmentState, dict]:
    """Advance SEIHRD state by one day."""
    S, E, I, H, R, D = state.S, state.E, state.I, state.H, state.R, state.D

    vaccinated = min(max(vaccines_used, 0.0) * params.vaccine_efficacy, S)
    S -= vaccinated
    R += vaccinated

    N_alive = S + E + I + H + R

    ppe_coverage = min(max(ppe_used, 0.0) / max(I, 1e-9), 1.0)
    ppe_factor = 1.0 - params.ppe_max_reduction * ppe_coverage

    beta = params.beta * beta_multiplier * ppe_factor
    new_infections = beta * S * I / max(N_alive, 1.0)
    new_infections = min(new_infections, S)

    new_infectious = params.sigma * E
    leaving_I = params.gamma * I
    new_hospitalizations = params.hosp_frac * leaving_I
    direct_recoveries = (1.0 - params.hosp_frac) * leaving_I

    vent_coverage = min(ventilators_used, H) / max(H, 1e-9)
    death_rate_per_H = vent_coverage * params.mu_vent + (1.0 - vent_coverage) * params.mu_no_vent
    deaths_from_H = death_rate_per_H * H
    leaving_H = params.gamma_h * H
    recoveries_from_H = max(leaving_H - deaths_from_H, 0.0)

    new_state = CompartmentState(
        S=max(S - new_infections, 0.0),
        E=max(E + new_infections - new_infectious, 0.0),
        I=max(I + new_infectious - leaving_I, 0.0),
        H=max(H + new_hospitalizations - leaving_H, 0.0),
        R=R + direct_recoveries + recoveries_from_H,
        D=D + deaths_from_H,
    )

    unmet_vent_demand = max(H - ventilators_used, 0.0)
    diagnostics = {
        "new_infections": new_infections,
        "new_hospitalizations": new_hospitalizations,
        "deaths_today": deaths_from_H,
        "unmet_vent_demand": unmet_vent_demand,
        "vent_coverage": vent_coverage,
        "vaccinated": vaccinated,
        "ppe_coverage": ppe_coverage,
    }
    return new_state, diagnostics
