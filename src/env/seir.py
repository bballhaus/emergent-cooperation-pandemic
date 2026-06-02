"""SEIHRD compartmental dynamics with ventilator-conditional mortality.

Compartments: S (susceptible), E (exposed), I (infectious), H (hospitalized/critical),
R (recovered), D (dead). Discrete daily timestep with Euler discretization.

Ventilator availability affects mortality among hospitalized: the fraction of H that
receives a ventilator dies at rate `mu_vent`; the unventilated fraction dies at rate
`mu_no_vent > mu_vent`. Total flux out of H is `gamma_h * H`; the death/recovery split
is determined by ventilator coverage.

Two further resources act through simplified channels (lightweight multi-resource model):
  - Vaccines move susceptibles directly to recovered/immune (S -> R), scaled by
    `vaccine_efficacy`. One dose immunizes one susceptible (capped at S).
  - PPE reduces effective transmission for the day: `ppe_used` units of coverage reduce
    beta multiplicatively up to `ppe_max_reduction`, with coverage measured against the
    infectious pool (the people whose contacts PPE has to interrupt).
Both default to no-op (0 used) so the ventilator-only path is unchanged.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SEIRParams:
    beta: float = 0.30          # transmission rate (per day); R0 = beta / gamma
    sigma: float = 1.0 / 4.0    # E -> I rate (1 / latent period; ~4 days for COVID-19)
    gamma: float = 1.0 / 8.0    # I -> (R or H) rate (1 / infectious period; ~8 days)
    hosp_frac: float = 0.025    # fraction of infections requiring critical care (~2.5%)
    gamma_h: float = 1.0 / 14.0 # H -> (R or D) rate (1 / average ICU stay; ~14 days)
    mu_no_vent: float = 0.90 / 14.0  # daily death rate in H without ventilator
    mu_vent: float = 0.40 / 14.0     # daily death rate in H with ventilator
    vaccine_efficacy: float = 0.9    # fraction of administered doses that immunize (S -> R)
    ppe_max_reduction: float = 0.5   # max multiplicative reduction of beta at full PPE coverage


@dataclass
class CompartmentState:
    """Continuous-valued compartment counts (we don't round to ints for stability)."""
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
    """Advance the SEIHRD state by one day. Returns new state and a diagnostics dict.

    `ventilators_used` is the number of currently-hospitalized patients receiving
    ventilation (capped at H by the caller). `beta_multiplier` lets the env apply a
    transient demand shock (e.g., 2x baseline during a surge window). `vaccines_used`
    immunizes susceptibles (S -> R) and `ppe_used` reduces effective beta for the day;
    both default to 0 (ventilator-only behavior unchanged).
    """
    S, E, I, H, R, D = state.S, state.E, state.I, state.H, state.R, state.D

    # Vaccination: doses move susceptibles to recovered/immune before mixing this day.
    vaccinated = min(max(vaccines_used, 0.0) * params.vaccine_efficacy, S)
    S -= vaccinated
    R += vaccinated

    N_alive = S + E + I + H + R  # exclude dead from mixing pool

    # PPE: coverage relative to the infectious pool reduces effective transmission.
    ppe_coverage = min(max(ppe_used, 0.0) / max(I, 1e-9), 1.0)
    ppe_factor = 1.0 - params.ppe_max_reduction * ppe_coverage

    beta = params.beta * beta_multiplier * ppe_factor
    new_infections = beta * S * I / max(N_alive, 1.0)
    new_infections = min(new_infections, S)  # cannot exceed susceptible pool

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
