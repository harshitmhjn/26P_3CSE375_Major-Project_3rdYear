"""Baseline schedulers (Phase-II report, Section 2.6).

All baselines expose the same ``act(env)`` interface as the learned agent
wrapper, so that the evaluation harness can run learned and heuristic policies
through identical code with no special-casing. This removes a common source of
unfair comparison.

Policies observe only what the environment exposes publicly: the action mask,
server allocations, server temperatures and the pending task. They do not look
at the future arrival stream.

Provided policies:

  RoundRobin        -- naive cyclic assignment; a floor reference.
  GreedyUtilisation -- lowest current utilisation; conventional utilisation-aware
                       scheduling (the "greedy / utilisation-based" baseline).
  GreedyCoolest     -- currently coolest feasible server; a simple thermal-aware
                       policy and the demanding baseline.
  RandomValid       -- uniform random among feasible servers; sanity reference.

When no server can accept the pending task (or none is pending) every policy
returns the defer action.

GreedyCoolest is the demanding baseline. The project's substantive claim is not
that reinforcement learning beats naive scheduling, but that an agent able to
anticipate heat accumulation improves on one that reacts to present temperature.
"""

import numpy as np


class BaselineScheduler:
    name = "baseline"

    def reset(self):
        pass

    def act(self, env):
        raise NotImplementedError


class RoundRobin(BaselineScheduler):
    name = "round-robin"

    def __init__(self):
        self._cursor = 0

    def reset(self):
        self._cursor = 0

    def act(self, env):
        mask = env.action_mask()
        n = env.n
        if env._pending() is None:
            return env.DEFER
        # Walk forward from the cursor to the next server with capacity.
        for offset in range(n):
            idx = (self._cursor + offset) % n
            if mask[idx]:
                self._cursor = (idx + 1) % n
                return idx
        return env.DEFER


class GreedyUtilisation(BaselineScheduler):
    name = "greedy-utilisation"

    def act(self, env):
        mask = env.action_mask()
        if env._pending() is None or not mask[: env.n].any():
            return env.DEFER
        util = env.alloc.copy()
        util[~mask[: env.n]] = np.inf
        return int(np.argmin(util))


class GreedyCoolest(BaselineScheduler):
    name = "greedy-coolest"

    def act(self, env):
        mask = env.action_mask()
        if env._pending() is None or not mask[: env.n].any():
            return env.DEFER
        temps = env.temps.copy()
        temps[~mask[: env.n]] = np.inf
        return int(np.argmin(temps))


class RandomValid(BaselineScheduler):
    """Uniform random choice among valid placements. Sanity-check reference."""

    name = "random"

    def __init__(self, seed=0):
        # One RNG stream for the policy's lifetime (it is deliberately NOT
        # re-seeded per episode). A freshly constructed RandomValid(seed)
        # evaluated over the same ordered seed list is fully deterministic,
        # and this is the behaviour behind the numbers in the Phase-II report.
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        mask = env.action_mask()
        if env._pending() is None or not mask[: env.n].any():
            return env.DEFER
        valid = np.flatnonzero(mask[: env.n])
        return int(self.rng.choice(valid))


BASELINES = {
    RoundRobin.name: RoundRobin,
    GreedyUtilisation.name: GreedyUtilisation,
    GreedyCoolest.name: GreedyCoolest,
    RandomValid.name: RandomValid,
}


def make_baseline(name: str):
    try:
        return BASELINES[name]()
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
