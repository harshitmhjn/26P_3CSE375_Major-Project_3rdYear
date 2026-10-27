"""Synthetic workload generator (Phase-II report, Section 4.4).

Each workload carries a CPU demand, a duration, and a thermal intensity
coefficient representing how much heat it generates per unit of utilisation --
distinguishing, for example, a compute-bound task from an I/O-bound one of
equivalent CPU demand.

Arrivals follow a Poisson process whose rate depends on the configured regime:

  steady  -- constant mean rate.
  diurnal -- rate varies sinusoidally, representing daily demand cycles.
  bursty  -- constant base rate with sharp transient spikes.

The generator owns its random number stream and is seeded, so identical trace
sequences are reproduced across runs and across policies *regardless of the
scheduling decisions taken*. This is what makes the controlled comparison in
Section 2.4 of the report possible.

Simplifying assumptions
-----------------------
* Only CPU is modelled as a resource; memory is not modelled.
* A task's CPU demand is constant for its whole lifetime.
* A task runs for exactly ``duration`` steps once placed. Tasks cannot be
  preempted or migrated (migration is out of scope; see the Phase-II report).
"""

from dataclasses import dataclass
import numpy as np

# Thermal-intensity threshold separating the two workload categories.
COMPUTE_BOUND_THRESHOLD = 1.0


@dataclass
class Task:
    task_id: int
    demand: float             # CPU demand, fraction of one server, in (0, 1]
    duration: int             # remaining duration in simulation steps
    thermal_intensity: float  # heat multiplier
    arrival_step: int
    total_duration: int = 0   # original duration (set on creation)
    start_step: int = -1      # step at which it was placed (-1 = not yet)
    finish_step: int = -1     # step at which it completed (-1 = not yet)
    server: int = -1          # server it was placed on (-1 = none)

    def __post_init__(self):
        if self.total_duration == 0:
            self.total_duration = self.duration

    @property
    def category(self) -> str:
        """Workload category derived from thermal intensity."""
        return ("compute-bound" if self.thermal_intensity >= COMPUTE_BOUND_THRESHOLD
                else "io-bound")

    def as_features(self):
        return (self.demand, self.duration, self.thermal_intensity)


class WorkloadGenerator:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self._next_id = 0
        self._burst_remaining = 0

    # -- arrival rate ------------------------------------------------------

    def _rate_at(self, step: int) -> float:
        cfg = self.cfg
        base = cfg.arrival_rate

        if cfg.regime == "steady":
            return base

        if cfg.regime == "diurnal":
            phase = 2.0 * np.pi * step / cfg.diurnal_period
            return base * (1.0 + cfg.diurnal_amplitude * np.sin(phase))

        if cfg.regime == "bursty":
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                return base * cfg.burst_multiplier
            if self.rng.random() < cfg.burst_prob:
                self._burst_remaining = cfg.burst_length - 1
                return base * cfg.burst_multiplier
            return base

        raise ValueError(f"unknown arrival regime: {cfg.regime!r}")

    # -- task creation -----------------------------------------------------

    def _make_task(self, step: int) -> Task:
        cfg = self.cfg
        task = Task(
            task_id=self._next_id,
            demand=float(self.rng.uniform(cfg.demand_min, cfg.demand_max)),
            duration=int(self.rng.integers(cfg.duration_min, cfg.duration_max + 1)),
            thermal_intensity=float(self.rng.uniform(cfg.thermal_min, cfg.thermal_max)),
            arrival_step=step,
        )
        self._next_id += 1
        return task

    def arrivals(self, step: int):
        """Sample the tasks arriving at this step."""
        rate = max(self._rate_at(step), 0.0)
        count = int(self.rng.poisson(rate))
        return [self._make_task(step) for _ in range(count)]
