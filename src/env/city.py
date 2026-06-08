"""City-level state: population, SEIR compartments, stockpiles, shock schedule."""

from dataclasses import dataclass, field
from typing import Optional

from .seir import SEIRParams, CompartmentState

PRIMARY_RESOURCE = "ventilator"


@dataclass
class CityConfig:
    name: str
    population: int
    hospital_capacity: int
    seir_params: SEIRParams = field(default_factory=SEIRParams)
    initial_infected: int = 10
    initial_stockpile: int = 0
    shock_start_day: int = 0
    shock_duration: int = 30
    shock_magnitude: float = 2.0


@dataclass
class IncomingTransfer:
    resource: str
    amount: int
    arrives_on_day: int
    sender_id: int


@dataclass
class City:
    config: CityConfig
    state: CompartmentState
    resources: list[str] = field(default_factory=lambda: [PRIMARY_RESOURCE])
    stockpiles: dict[str, int] = field(default_factory=dict)
    incoming: list[IncomingTransfer] = field(default_factory=list)

    cumulative_deaths: float = 0.0
    cumulative_unmet_vent_days: float = 0.0
    cumulative_sent: int = 0
    cumulative_received: int = 0
    transfers_received_this_step: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: CityConfig, resources: Optional[list[str]] = None) -> "City":
        resources = list(resources) if resources else [PRIMARY_RESOURCE]
        return cls(
            config=cfg,
            state=CompartmentState.initial(cfg.population, cfg.initial_infected),
            resources=resources,
            stockpiles={r: (cfg.initial_stockpile if r == PRIMARY_RESOURCE else 0) for r in resources},
        )

    @property
    def stockpile(self) -> int:
        return self.stockpiles.get(PRIMARY_RESOURCE, 0)

    @stockpile.setter
    def stockpile(self, value: int) -> None:
        self.stockpiles[PRIMARY_RESOURCE] = int(value)

    def reset(self) -> None:
        self.state = CompartmentState.initial(self.config.population, self.config.initial_infected)
        self.stockpiles = {
            r: (self.config.initial_stockpile if r == PRIMARY_RESOURCE else 0) for r in self.resources
        }
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
        """Deposit due transfers into stockpiles."""
        self.transfers_received_this_step = {}
        remaining: list[IncomingTransfer] = []
        for t in self.incoming:
            if t.arrives_on_day <= day:
                self.stockpiles[t.resource] = self.stockpiles.get(t.resource, 0) + t.amount
                self.cumulative_received += t.amount
                if t.resource == PRIMARY_RESOURCE:
                    self.transfers_received_this_step[t.sender_id] = (
                        self.transfers_received_this_step.get(t.sender_id, 0) + t.amount
                    )
            else:
                remaining.append(t)
        self.incoming = remaining

    def send(
        self,
        amount: int,
        recipient: "City",
        sender_id: int,
        transit_days: int,
        day: int,
        resource: str = PRIMARY_RESOURCE,
    ) -> int:
        """Send resource units to recipient."""
        amount = max(0, min(amount, self.stockpiles.get(resource, 0)))
        if amount == 0:
            return 0
        self.stockpiles[resource] -= amount
        self.cumulative_sent += amount
        recipient.incoming.append(
            IncomingTransfer(
                resource=resource,
                amount=amount,
                arrives_on_day=day + transit_days,
                sender_id=sender_id,
            )
        )
        return amount
