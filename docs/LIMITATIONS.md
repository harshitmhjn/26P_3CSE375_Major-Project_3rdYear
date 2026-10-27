# Assumptions, limitations, future work

## Assumptions
* Lumped-parameter thermal dynamics with a fixed rack-aware coupling matrix.
* Task demand/duration/thermal intensity are drawn from simple distributions.
* Homogeneous servers; one task needs one server's capacity slice.
* Cooling energy is a proxy; no CRAC, airflow, or PUE model.

## Limitations
* **Simulation only.** Nothing was validated against real telemetry, CFD, or hardware.
* Thermal parameters are plausible, not calibrated.
* DRL agents were trained for a limited budget (300k steps, 3 seeds per
  algorithm) on the steady regime only. See RESULTS.md for how they performed;
  they should not be assumed to be better than heuristics.
* DQN is prone to collapsing to a degenerate always-defer policy in this setup
  (reward shaping trade-off); low hotspot numbers from such a run are not a good result.
* Hyper-parameters were not extensively tuned. Any tuning against the held-out
  seeds would contaminate the comparison and is called out where it occurs.
* Dashboard "live" simulation is single-process and not intended for large N.

## Future work
* Calibrate thermal model to measured data; add heterogeneous servers.
* Longer training, hyper-parameter search using a separate validation seed set.
* Action masking inside training (e.g. MaskablePPO), reward-weight sweeps.
* Real traces (e.g. Google/Alibaba cluster traces) as workload.
* Multi-task placement, migration, cooling-setpoint control.
