# Status, pending work and evidence list

This file reports technical status only. It contains **no** weekly progress reports,
supervisor approvals or institutional submissions - those are for the team to prepare
and to coordinate with the guide.

## Complete (verified by running the code here)
* Simulator, workload generator, thermal and energy models, config validation.
* Baselines: round-robin, greedy-utilisation, greedy-coolest, random-valid.
* Gymnasium env; PPO and DQN training with checkpoints; 3 training seeds each at 300k steps (all 6 `complete`).
* Evaluation on held-out seeds for 3 regimes; CSVs, summary tables, figures, auto report.
* Streamlit dashboard (6 pages), tested with Streamlit AppTest; pytest suite (see TESTING.md).

## Not achieved / honest caveats
* DRL does not beat greedy-coolest; DQN partly collapses to deferring work.
* No hyper-parameter tuning, no reward-weight sweep, 3 training seeds only.
* Thermal parameters are not calibrated to real hardware; cooling energy is a proxy.
* Dashboard was exercised via AppTest and a headless server start, not by manual browser testing on Windows.
* Windows setup commands were written but **not run on a Windows machine** by me.

## Pending (team decisions)
* Decide whether to spend more effort on DRL improvement (longer training, masked PPO, reward weights) - use a *separate validation seed set*, not 1001-1005.
* Align existing documents with this state: the Week-4 WPR text ("environment stubbed") and Phase-II report section 9.4 ("training in progress") and any older 20k-step PPO numbers.
* Weekly reports, supervisor review and the Google Form submission (due 29-Oct-2026).

## Evidence for the Google Form (technical evidence you can link)
| Item | Where |
|---|---|
| Source code | this repository / ZIP (push to GitHub and link it) |
| Architecture diagram | `docs/architecture.png`, `docs/architecture.dot` |
| Method | `docs/METHODOLOGY.md` |
| Test evidence | `docs/TESTING.md`, `results/logs/pytest_full.log` |
| Training logs and curves | `results/logs/train_*.log`, `results/models/*_progress.json`, `*_curve.npz` |
| Experimental results | `results/experiments/<regime>/` (`raw_results.csv`, `summary.csv`, `config.json`, `report.md`, `figures/`), `docs/RESULTS.md` |
| Demo | screen-record `streamlit run app.py` (you must record this yourself) |
| Limitations | `docs/LIMITATIONS.md` |
