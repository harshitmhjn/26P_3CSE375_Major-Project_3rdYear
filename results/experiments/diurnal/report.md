# Automated evaluation report - `diurnal`

* Arrival regime: **diurnal**, episode length **1000** steps, **16** servers in **4** racks.
* Held-out evaluation seeds: [1001, 1002, 1003, 1004, 1005] (identical workload traces for every policy).
* Reward weights: alpha=1.0, beta=0.5, gamma_w=0.3.
* Trained DRL runs evaluated: DQN-seed0, DQN-seed1, DQN-seed2, PPO-seed0, PPO-seed1, PPO-seed2.

## Summary (mean +/- s.d. across held-out seeds)

| Policy | Hotspot freq. | Peak T (C) | Mean T (C) | IT energy (kW*step) | Cooling proxy (kW*step) | Wait (steps) | Response (steps) | Completed | Return |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.463 +/- 0.042 | 65.79 +/- 0.92 | 47.63 +/- 0.44 | 9351 +/- 170 | 3758 +/- 58 | 5.64 +/- 0.43 | 41.8 +/- 0.7 | 774.6 +/- 14.4 | -2150 +/- 288 |
| round-robin | 0.355 +/- 0.053 | 64.36 +/- 0.90 | 47.65 +/- 0.44 | 9751 +/- 127 | 3563 +/- 64 | 5.64 +/- 0.43 | 41.8 +/- 0.7 | 774.6 +/- 14.4 | -1778 +/- 256 |
| greedy-utilisation | 0.201 +/- 0.062 | 63.26 +/- 0.39 | 47.63 +/- 0.44 | 9836 +/- 128 | 3387 +/- 51 | 5.66 +/- 0.44 | 41.8 +/- 0.7 | 773.6 +/- 13.6 | -1410 +/- 151 |
| greedy-coolest | 0.172 +/- 0.069 | 61.59 +/- 1.26 | 47.59 +/- 0.45 | 9728 +/- 128 | 3378 +/- 73 | 5.64 +/- 0.43 | 41.8 +/- 0.7 | 774.4 +/- 14.3 | -1296 +/- 142 |
| PPO | 0.317 +/- 0.036 | 64.18 +/- 0.44 | 46.71 +/- 0.38 | 8574 +/- 129 | 3656 +/- 48 | 6.41 +/- 0.41 | 42.6 +/- 0.7 | 742.9 +/- 17.6 | -1685 +/- 147 |
| DQN | 0.104 +/- 0.033 | 51.27 +/- 0.44 | 38.76 +/- 0.12 | 4437 +/- 33 | 2713 +/- 23 | 7.91 +/- 0.30 | 32.1 +/- 0.6 | 386.7 +/- 7.0 | -1940 +/- 87 |

## Findings

* Best heuristic on hotspot frequency: **greedy-coolest** (0.172).
* **PPO** hotspot frequency 0.317 is WORSE than greedy-coolest (+0.144); episode return -389; tasks completed -31.5.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.042 (s.d.).
  * Caution: PPO completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.
* **DQN** hotspot frequency 0.104 is BETTER than greedy-coolest (-0.068); episode return -643; tasks completed -387.7.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.079 (s.d.).
  * Caution: DQN completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.

## Notes

* Energy: IT energy = sum over steps of per-server power (idle + dynamic x utilisation). The cooling figure is a **proxy** (coefficient x hottest-server excess over ambient), not a facility cooling model; do not read it as a measured cooling saving.
* All figures were produced by this pipeline from `raw_results.csv`; none is hardcoded.
* The thermal model is a simplified lumped-parameter approximation, not CFD, and is not validated against a physical facility.
