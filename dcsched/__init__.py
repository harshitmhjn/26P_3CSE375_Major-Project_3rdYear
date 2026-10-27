"""Thermal-aware data-centre workload scheduling with deep reinforcement learning.

Major Project, B.Tech. CSE, IILM University Greater Noida, Session 2026-27.

The core package (config, thermal, workload, env, baselines) depends only on
numpy and gymnasium. Learned-agent code (agent, training) imports torch and
stable-baselines3 lazily, so the simulator and baselines run without them.
"""

from .config import (
    EnvConfig, FacilityConfig, ThermalConfig, WorkloadConfig, RewardConfig,
)
from .env import DataCenterSchedulingEnv
from .thermal import ThermalModel
from .workload import WorkloadGenerator, Task
from .baselines import (
    RoundRobin, GreedyUtilisation, GreedyCoolest, RandomValid, BASELINES,
    make_baseline,
)

__all__ = [
    "EnvConfig", "FacilityConfig", "ThermalConfig", "WorkloadConfig",
    "RewardConfig", "DataCenterSchedulingEnv", "ThermalModel",
    "WorkloadGenerator", "Task", "RoundRobin", "GreedyUtilisation",
    "GreedyCoolest", "RandomValid", "BASELINES", "make_baseline",
]
