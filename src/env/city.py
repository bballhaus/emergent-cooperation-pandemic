"""City-level state: population, SEIR compartments, stockpile, shock schedule."""

from dataclasses import dataclass, field
from typing import Optional

from .seir import SEIRParams, CompartmentState


@dataclass
class CityConfig:
    name: str
    population: int
    hospital_capacity: int           # baseline ventilator-capable beds (used only for obs scaling)
    seir_params: SEIRParams = field(default_factory=SEIRParams)
    initial_infected: int = 10
    initial_stockpile: int = 0       # ventilators on hand at t=0
    shock_start_day: int = 0         # day a demand surge begins
    shock_duration: int = 30         # days the surge lasts
    shock_magnitude: float = 2.0     # multiplier on beta during surge


@dataclass
class IncomingTransfer:
    resource: str         # "ventilator" for now; vaccines/PPE later
    amount: int
    arrives_on_day: int
    sender_id: int        # index of the sending city (for peer-incentive bookkeeping)


@dataclass
class City:
    config: CityConfig
    state: CompartmentState
    stockpile: int = 0
    incoming: list[IncomingTransfer] = field(default_factory=list)

    cumulative_deaths: float = 0.0
    cumulative_unmet_vent_days: float = 0.0
    cumulative_sent: int = 0
    cumulative_received: int = 0
    # transfers_received_this_step[sender_id] = amount delivered this step; cleared each step
    transfers_received_this_step: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: CityConfig) -> "City":
        return cls(
            config=cfg,
            state=CompartmentState.initial(cfg.population, cfg.initial_infected),
            stockpile=cfg.initial_stockpile,
        )

    def reset(self) -> None:
        self.state = CompartmentState.initial(self.config.population, self.config.initial_infected)
        self.stockpile = self.config.initial_stockpile
        self.incoming = []
        self.cumulative_deaths = 0.0
        self.cumulative_unmet_vent_days = 0.0
        self.cumulative_sent = 0
        self.cumulative_received = 0
        self.transfers_received_this_step = {}

    def beta_multiplier(self, day: int) -> float:
        cfg = self.config
        if cfg.shock_start_day <= day < cfg.shock_start_day + cfg.shock_duration:
            return cfg.shock_magnitude
        return 1.0

    def receive_arrivals(self, day: int) -> None:
        """Move any transfers whose arrival day has come into the stockpile."""
        self.transfers_received_this_step = {}
        remaining: list[IncomingTransfer] = []
        for t in self.incoming:
            if t.arrives_on_day <= day:
                self.stockpile += t.amount
                self.cumulative_received += t.amount
                self.transfers_received_this_step[t.sender_id] = (
                    self.transfers_received_this_step.get(t.sender_id, 0) + t.amount
                )
            else:
                remaining.append(t)
        self.incoming = remaining

    def send(self, amount: int, recipient: "City", sender_id: int, transit_days: int, day: int) -> int:
        """Send `amount` ventilators to `recipient`. Returns amount actually sent (capped at stockpile)."""
        amount = max(0, min(amount, self.stockpile))
        if amount == 0:
            return 0
        self.stockpile -= amount
        self.cumulative_sent += amount
        recipient.incoming.append(
            IncomingTransfer(
                resource="ventilator",
                amount=amount,
                arrives_on_day=day + transit_days,
                sender_id=sender_id,
            )
        )
        return amount
