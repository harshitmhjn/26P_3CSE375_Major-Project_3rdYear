"""Configuration objects for the thermal-aware scheduling simulator.

All tunable parameters live here rather than being embedded in the modules, so
that any reported result can be reproduced from its configuration alone
(Phase-II report, Section 4.7). Configurations round-trip through JSON.

Units
-----
Temperatures      degrees Celsius.
Time              one simulation step (the model's fixed time step, dt = 1).
CPU demand        fraction of one server's CPU capacity (1.0 = a full server).
Power             kW (idle_power / dynamic_power). Energy is kW * step.
"""

from dataclasses import dataclass, asdict, field
import json


@dataclass
class FacilityConfig:
    """Physical layout and capacity of the simulated data centre."""

    n_servers: int = 16
    n_racks: int = 4
    # Normalised CPU capacity per server. 1.0 == fully utilised.
    server_capacity: float = 1.0

    def __post_init__(self):
        if self.n_servers < 1 or self.n_racks < 1:
            raise ValueError("n_servers and n_racks must be >= 1")
        if self.n_servers % self.n_racks != 0:
            raise ValueError("n_servers must be divisible by n_racks")
        if self.server_capacity <= 0:
            raise ValueError("server_capacity must be > 0")

    @property
    def servers_per_rack(self) -> int:
        return self.n_servers // self.n_racks

    def rack_of(self, server_index: int) -> int:
        return server_index // self.servers_per_rack


@dataclass
class ThermalConfig:
    """Lumped-parameter thermal model coefficients (Phase-II report, Section 4.3).

    Temperatures are in degrees Celsius. The model is a simplified approximation
    tuned to reproduce the qualitative behaviours that make thermal-aware
    scheduling non-trivial; it is NOT a CFD model and has NOT been validated
    against a physical facility.
    """

    ambient_temp: float = 22.0
    idle_temp: float = 30.0          # steady-state temperature of an idle server
    # Heat generated per unit of utilisation per step (deg C / step).
    heat_per_util: float = 4.0
    # Fraction of a neighbour's excess heat received per step.
    recirc_same_rack: float = 0.030
    recirc_cross_rack: float = 0.004
    # Rate at which a server relaxes back toward its idle temperature.
    cooling_rate: float = 0.22
    # Safe operating threshold; exceeding this counts as a hotspot.
    safe_temp: float = 58.0
    # Hard failure threshold; exceeding this terminates the episode.
    critical_temp: float = 75.0
    # Normalisation divisor for temperature features in the observation.
    temp_norm: float = 58.0


@dataclass
class WorkloadConfig:
    """Synthetic workload generator parameters (Phase-II report, Section 4.4)."""

    # Mean arrivals per step (Poisson rate).
    arrival_rate: float = 0.9
    # CPU demand is sampled uniformly from this range.
    demand_min: float = 0.10
    demand_max: float = 0.45
    # Duration in simulation steps.
    duration_min: int = 12
    duration_max: int = 60
    # Thermal intensity: heat produced per unit utilisation, relative to 1.0.
    # Distinguishes a compute-bound task from an I/O-bound one.
    thermal_min: float = 0.55
    thermal_max: float = 1.45
    # Arrival regime: "steady", "diurnal" or "bursty".
    regime: str = "steady"
    # Diurnal regime: rate oscillates by +/- this fraction over the period.
    diurnal_amplitude: float = 0.6
    diurnal_period: int = 250
    # Bursty regime: probability a burst starts on a given step, and its scale.
    burst_prob: float = 0.012
    burst_multiplier: float = 5.0
    burst_length: int = 18
    # Maximum tasks held in the pending queue.
    queue_capacity: int = 12

    def __post_init__(self):
        if self.regime not in ("steady", "diurnal", "bursty"):
            raise ValueError(f"unknown arrival regime: {self.regime!r}")
        if not (0 < self.demand_min <= self.demand_max <= 1.0):
            raise ValueError("require 0 < demand_min <= demand_max <= 1")
        if not (1 <= self.duration_min <= self.duration_max):
            raise ValueError("require 1 <= duration_min <= duration_max")
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate must be >= 0")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")


@dataclass
class RewardConfig:
    """Reward weighting (Phase-II report, Section 3.3).

    alpha weights thermal risk, beta energy cost, gamma_w performance penalty.
    """

    alpha: float = 1.0
    beta: float = 0.5
    gamma_w: float = 0.3
    # Fixed penalty applied whenever any server is above the safe threshold.
    hotspot_penalty: float = 1.0
    # Penalty for choosing the defer action.
    defer_penalty: float = 0.35
    # Penalty per step per task waiting in the queue.
    wait_penalty: float = 0.06
    # Penalty for each task lost to queue overflow. Without this, an agent can
    # avoid all thermal risk by deferring until the queue overflows and the
    # work disappears at no cost -- a degenerate policy that scores well on
    # every thermal metric while completing half the workload.
    drop_penalty: float = 8.0
    # Penalty when the agent selects an action masked as infeasible (the
    # environment executes it as a defer). Baselines never trigger this.
    invalid_penalty: float = 0.5
    # Penalty applied on episode termination by critical overheat.
    critical_penalty: float = 60.0
    # IT power model: P = idle_power (if the server hosts any task)
    #                   + dynamic_power * utilisation          [kW]
    idle_power: float = 0.30
    dynamic_power: float = 0.70
    # Cooling power PROXY: cooling_coeff * (hottest server - ambient)   [kW]
    cooling_coeff: float = 0.11
    # Divisor applied to the raw energy term to keep reward components
    # on comparable scales.
    energy_scale: float = 10.0


@dataclass
class EnvConfig:
    """Top-level environment configuration."""

    facility: FacilityConfig = field(default_factory=FacilityConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode_length: int = 1000
    seed: int = 0

    def __post_init__(self):
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def from_dict(d):
        return EnvConfig(
            facility=FacilityConfig(**d.get("facility", {})),
            thermal=ThermalConfig(**d.get("thermal", {})),
            workload=WorkloadConfig(**d.get("workload", {})),
            reward=RewardConfig(**d.get("reward", {})),
            episode_length=d.get("episode_length", 1000),
            seed=d.get("seed", 0),
        )

    @staticmethod
    def from_json(path):
        with open(path) as fh:
            return EnvConfig.from_dict(json.load(fh))
