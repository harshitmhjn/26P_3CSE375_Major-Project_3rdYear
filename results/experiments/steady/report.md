# Automated evaluation report - `steady`

* Arrival regime: **steady**, episode length **1000** steps, **16** servers in **4** racks.
* Held-out evaluation seeds: [1001, 1002, 1003, 1004, 1005] (identical workload traces for every policy).
* Reward weights: alpha=1.0, beta=0.5, gamma_w=0.3.
* Trained DRL runs evaluated: DQN-seed0, DQN-seed1, DQN-seed2, PPO-seed0, PPO-seed1, PPO-seed2.

## Summary (mean +/- s.d. across held-out seeds)

| Policy | Hotspot freq. | Peak T (C) | Mean T (C) | IT energy (kW*step) | Cooling proxy (kW*step) | Wait (steps) | Response (steps) | Completed | Return |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.593 +/- 0.048 | 66.19 +/- 2.45 | 50.19 +/- 0.53 | 10422 +/- 207 | 3976 +/- 68 | 4.04 +/- 0.78 | 40.1 +/- 0.7 | 876.8 +/- 20.8 | -2426 +/- 749 |
| round-robin | 0.483 +/- 0.056 | 64.86 +/- 1.59 | 50.19 +/- 0.55 | 10723 +/- 175 | 3866 +/- 70 | 4.04 +/- 0.78 | 40.1 +/- 0.7 | 876.8 +/- 20.8 | -1952 +/- 531 |
| greedy-utilisation | 0.280 +/- 0.079 | 63.97 +/- 1.45 | 50.19 +/- 0.54 | 10772 +/- 158 | 3671 +/- 57 | 4.05 +/- 0.78 | 40.2 +/- 0.7 | 876.8 +/- 20.8 | -1341 +/- 316 |
| greedy-coolest | 0.245 +/- 0.071 | 63.21 +/- 2.25 | 50.13 +/- 0.54 | 10698 +/- 163 | 3649 +/- 76 | 4.04 +/- 0.78 | 40.1 +/- 0.7 | 876.8 +/- 20.8 | -1299 +/- 323 |
| PPO | 0.472 +/- 0.048 | 64.44 +/- 0.78 | 49.07 +/- 0.58 | 9580 +/- 229 | 3865 +/- 51 | 5.84 +/- 0.60 | 41.9 +/- 0.7 | 839.7 +/- 23.0 | -1838 +/- 244 |
| DQN | 0.125 +/- 0.031 | 51.36 +/- 1.09 | 39.14 +/- 0.11 | 4571 +/- 27 | 2757 +/- 21 | 10.47 +/- 0.44 | 34.5 +/- 0.6 | 398.3 +/- 4.9 | -1980 +/- 55 |

## Findings

* Best heuristic on hotspot frequency: **greedy-coolest** (0.245).
* **PPO** hotspot frequency 0.472 is WORSE than greedy-coolest (+0.227); episode return -539; tasks completed -37.1.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.063 (s.d.).
  * Caution: PPO completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.
* **DQN** hotspot frequency 0.125 is BETTER than greedy-coolest (-0.120); episode return -681; tasks completed -478.5.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.090 (s.d.).
  * Caution: DQN completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.

## Notes

* Energy: IT energy = sum over steps of per-server power (idle + dynamic x utilisation). The cooling figure is a **proxy** (coefficient x hottest-server excess over ambient), not a facility cooling model; do not read it as a measured cooling saving.
* All figures were produced by this pipeline from `raw_results.csv`; none is hardcoded.
* The thermal model is a simplified lumped-parameter approximation, not CFD, and is not validated against a physical facility.
