# Automated evaluation report - `bursty`

* Arrival regime: **bursty**, episode length **1000** steps, **16** servers in **4** racks.
* Held-out evaluation seeds: [1001, 1002, 1003, 1004, 1005] (identical workload traces for every policy).
* Reward weights: alpha=1.0, beta=0.5, gamma_w=0.3.
* Trained DRL runs evaluated: DQN-seed0, DQN-seed1, DQN-seed2, PPO-seed0, PPO-seed1, PPO-seed2.

## Summary (mean +/- s.d. across held-out seeds)

| Policy | Hotspot freq. | Peak T (C) | Mean T (C) | IT energy (kW*step) | Cooling proxy (kW*step) | Wait (steps) | Response (steps) | Completed | Return |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.603 +/- 0.048 | 65.85 +/- 1.03 | 50.30 +/- 0.61 | 10523 +/- 221 | 3990 +/- 49 | 5.95 +/- 1.12 | 41.4 +/- 1.1 | 903.4 +/- 25.8 | -3681 +/- 365 |
| round-robin | 0.509 +/- 0.063 | 66.49 +/- 0.98 | 50.30 +/- 0.62 | 10800 +/- 184 | 3886 +/- 62 | 5.95 +/- 1.12 | 41.4 +/- 1.1 | 903.4 +/- 25.8 | -3425 +/- 359 |
| greedy-utilisation | 0.292 +/- 0.065 | 64.19 +/- 0.91 | 50.27 +/- 0.61 | 10833 +/- 180 | 3699 +/- 63 | 6.03 +/- 1.13 | 41.5 +/- 1.1 | 901.4 +/- 25.5 | -2718 +/- 424 |
| greedy-coolest | 0.223 +/- 0.047 | 64.15 +/- 2.00 | 50.24 +/- 0.62 | 10769 +/- 188 | 3647 +/- 64 | 5.95 +/- 1.12 | 41.4 +/- 1.1 | 903.2 +/- 25.7 | -2595 +/- 405 |
| PPO | 0.453 +/- 0.044 | 63.99 +/- 0.26 | 48.92 +/- 0.54 | 9542 +/- 196 | 3863 +/- 61 | 7.62 +/- 0.57 | 43.0 +/- 0.6 | 853.1 +/- 24.9 | -3022 +/- 451 |
| DQN | 0.123 +/- 0.050 | 51.19 +/- 0.49 | 39.13 +/- 0.13 | 4579 +/- 25 | 2751 +/- 38 | 10.23 +/- 0.36 | 33.9 +/- 0.4 | 408.5 +/- 8.9 | -3312 +/- 418 |

## Findings

* Best heuristic on hotspot frequency: **greedy-coolest** (0.223).
* **PPO** hotspot frequency 0.453 is WORSE than greedy-coolest (+0.230); episode return -427; tasks completed -50.1.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.082 (s.d.).
  * Caution: PPO completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.
* **DQN** hotspot frequency 0.123 is BETTER than greedy-coolest (-0.100); episode return -718; tasks completed -494.7.
  * Aggregated over 3 training seed(s); spread of hotspot frequency across training seeds: 0.087 (s.d.).
  * Caution: DQN completes fewer tasks than greedy-coolest; check for a degenerate deferral policy before reading its thermal numbers favourably.

## Notes

* Energy: IT energy = sum over steps of per-server power (idle + dynamic x utilisation). The cooling figure is a **proxy** (coefficient x hottest-server excess over ambient), not a facility cooling model; do not read it as a measured cooling saving.
* All figures were produced by this pipeline from `raw_results.csv`; none is hardcoded.
* The thermal model is a simplified lumped-parameter approximation, not CFD, and is not validated against a physical facility.
