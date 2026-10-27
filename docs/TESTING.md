# Testing

Run: `python -m pytest -q`. Last full run in this workspace (Linux, Python 3.13, numpy 2.5.3,
torch 2.14.1, stable-baselines3 2.9.0, gymnasium 1.4.0, streamlit 1.66.1):
**116 passed, 2 warnings, 40 s** (log: `results/logs/pytest_full.log`). Not run on Windows.

| Test file | Tests |
|---|---|
| `test_app.py` | 16 |
| `test_baselines.py` | 17 |
| `test_env.py` | 18 |
| `test_fairness_metrics.py` | 12 |
| `test_live_policies.py` | 7 |
| `test_pipeline.py` | 16 |
| `test_thermal_energy.py` | 12 |
| `test_workload.py` | 18 |

## Coverage of the required areas
* Workload generation: determinism, regimes, seed independence (`test_workload`, `test_fairness_metrics`).
* Server capacity: no over-allocation, invalid actions penalised/masked (`test_env`, `test_baselines`).
* Task lifecycle: arrival, queue overflow drop, completion, response = wait + duration (`test_env`).
* Thermal and energy: stability check, heating/cooling direction, recirculation, power model (`test_thermal_energy`).
* Baselines: valid actions, round-robin order, greedy choices (`test_baselines`).
* Env interface: Gymnasium checker, spaces, reset/step determinism (`test_env`).
* Checkpoints: save/load/resume, incomplete runs excluded, incompatible size rejected (`test_pipeline`, `test_live_policies`).
* Metrics: held-out seed disjointness, no future leakage (poisoned generator), always-defer guard (`test_fairness_metrics`).
* Experiment export: CSV/summary/config/report files written (`test_pipeline`).
* UI launch: Streamlit AppTest renders all 6 pages, buttons/config/training launch (`test_app`).
* Invalid inputs and missing checkpoints: config validation, clear errors (`test_env`, `test_live_policies`, `test_app`).

## Limits of this evidence
Training-pipeline tests use tiny runs (they prove the code works, not that agents are good).
UI tests use AppTest, not a human clicking in a browser.
