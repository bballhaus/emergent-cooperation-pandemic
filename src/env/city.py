"""City-level state: population, SEIR compartments, stockpiles, shock schedule."""

from dataclasses import dataclass, field
from typing import Optional

from .seir import SEIRParams, CompartmentState

# The primary transferable resource. Impact/peer cooperation credit is scored on this
# resource only (its shortfall = H - on-hand), so the cooperation mechanism is unchanged
# from the single-resource study; vaccines/PPE are shared but not specially shaped.
PRIMARY_RESOURCE = "ventilator"


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
    resource: str         # one of the env's resources (e.g. "ventilator", "vaccine", "ppe")
    amount: int
    arrives_on_day: int
    sender_id: int        # index of the sending city (for peer-incentive bookkeeping)


@dataclass
class City:
    config: CityConfig
    state: CompartmentState
    resources: list[str] = field(default_factory=lambda: [PRIMARY_RESOURCE])
    stockpiles: dict[str, int] = field(default_factory=dict)
    incoming: list[IncomingTransfer] = field(default_factory=list)

    cumulative_deaths: float = 0.0
    cumulative_unmet_vent_days: float = 0.0
    cumulative_sent: int = 0          # total units sent across all resources (cooperation volume)
    cumulative_received: int = 0      # total units received across all resources
    # transfers_received_this_step[sender_id] = PRIMARY-resource units delivered this step
    # (cleared each step). Scoped to the primary resource so impact/peer credit is unchanged.
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

    # Backward-compatible scalar view of the primary-resource stockpile.
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
        """Move any transfers whose arrival day has come into the matching resource stockpile."""
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
        """Send `amount` units of `resource` to `recipient`. Returns amount actually sent."""
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
