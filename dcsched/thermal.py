"""Lumped-parameter thermal model (Phase-II report, Section 4.3).

Each server's temperature evolves under three influences:

  1. Heat generated in proportion to its own current utilisation, weighted by
     the thermal intensity of the workloads it is running.
  2. Heat received from other servers through recirculation, weighted more
     strongly within a rack than between racks.
  3. Relaxation toward the server's idle temperature (cooling).

Update rule (one step, dt = 1):

    T_i(t+1) = T_i(t)
               + heat_per_util * u_i * k_i                    (self heating)
               + sum_j C_ij * max(T_j(t) - T_idle, 0)         (recirculation)
               - cooling_rate * (T_i(t) - T_idle)             (cooling)

and T_i is floored at the ambient temperature. Recirculation is measured as
excess above the idle temperature, which makes the all-idle state an exact
fixed point of the model.

The model is deliberately simple. Its purpose is to reproduce the qualitative
behaviours that make thermal-aware scheduling non-trivial -- heat accumulates
over time rather than instantaneously, adjacent servers influence one another,
and concentrated load produces localised hotspots. No claim is made that its
absolute temperature predictions match a physical facility.

It is kept behind a narrow interface (``step``) so that it can be replaced with
a higher-fidelity model without changes elsewhere.
"""

import numpy as np


class ThermalModel:
    def __init__(self, facility_cfg, thermal_cfg):
        self.fac = facility_cfg
        self.cfg = thermal_cfg
        self.n = facility_cfg.n_servers
        self._coupling = self._build_coupling_matrix()
        self._check_stability()

    def _build_coupling_matrix(self) -> np.ndarray:
        """Heat recirculation coefficients between every pair of servers.

        Entry (i, j) is the fraction of server j's excess heat that reaches
        server i per step. Servers in the same rack are coupled more strongly
        than servers in different racks. The diagonal is zero -- a server's own
        heat is handled by the generation term, not by recirculation.
        """
        n = self.n
        m = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                same_rack = self.fac.rack_of(i) == self.fac.rack_of(j)
                if same_rack:
                    # Within a rack, coupling weakens with vertical distance.
                    distance = abs(i - j)
                    m[i, j] = self.cfg.recirc_same_rack / distance
                else:
                    m[i, j] = self.cfg.recirc_cross_rack
        return m

    def _check_stability(self):
        """Verify that recirculation cannot drive unbounded temperature growth.

        Each server receives ``sum_j coupling[i, j] * excess_j`` per step and
        sheds ``cooling_rate * excess_i``. If the largest row sum of the
        coupling matrix meets or exceeds the cooling rate, heat fed back into
        the facility outpaces heat removed from it and temperatures diverge
        regardless of how workloads are scheduled -- which would make every
        scheduling policy look identically catastrophic.
        """
        gain = float(self._coupling.sum(axis=1).max())
        if gain >= self.cfg.cooling_rate:
            raise ValueError(
                f"unstable thermal configuration: peak recirculation gain "
                f"{gain:.4f} >= cooling rate {self.cfg.cooling_rate:.4f}. "
                f"Reduce recirc_same_rack/recirc_cross_rack or raise cooling_rate."
            )
        self.recirculation_gain = gain

    def initial_temperatures(self) -> np.ndarray:
        return np.full(self.n, self.cfg.idle_temp, dtype=np.float64)

    def step(self, temps: np.ndarray, utilisation: np.ndarray,
             thermal_intensity: np.ndarray) -> np.ndarray:
        """Advance temperatures by one simulation step.

        Parameters
        ----------
        temps : current temperature of each server (deg C).
        utilisation : current CPU utilisation of each server, in [0, 1].
        thermal_intensity : effective heat multiplier per server, reflecting
            the mix of workloads it is currently running.

        Returns
        -------
        Updated temperature array.
        """
        generated = self.cfg.heat_per_util * utilisation * thermal_intensity

        excess = np.maximum(temps - self.cfg.idle_temp, 0.0)
        received = self._coupling @ excess

        cooling = self.cfg.cooling_rate * (temps - self.cfg.idle_temp)

        new_temps = temps + generated + received - cooling
        return np.clip(new_temps, self.cfg.ambient_temp, None)

    # -- convenience predicates -------------------------------------------

    def hotspots(self, temps: np.ndarray) -> np.ndarray:
        return temps > self.cfg.safe_temp

    def any_critical(self, temps: np.ndarray) -> bool:
        return bool(np.any(temps > self.cfg.critical_temp))
