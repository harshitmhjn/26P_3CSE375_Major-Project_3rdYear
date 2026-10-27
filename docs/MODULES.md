# Module descriptions

| File | Purpose |
|---|---|
| `dcsched/config.py` | Validated dataclass configs (facility, thermal, workload, reward); JSON round-trip. |
| `dcsched/workload.py` | Seeded task generator (steady/diurnal/bursty), `Task` record. |
| `dcsched/thermal.py` | Rack-aware thermal coupling matrix and temperature update; stability check. |
| `dcsched/env.py` | Gymnasium environment, reward, per-episode metric accumulator, snapshots for the UI. |
| `dcsched/baselines.py` | Round-robin, greedy-utilisation, greedy-coolest, random-valid policies. |
| `dcsched/agent.py` | SB3 PPO/DQN construction, loading, `PolicyAgent` wrapper with masked fallback. |
| `dcsched/training.py` | Resumable training, checkpoints, curves, held-out evaluation record. |
| `dcsched/evaluate.py` | Episode runner, multi-seed evaluation, comparison and headline tables. |
| `dcsched/experiments.py` | End-to-end experiment: CSVs, summary, config record, figures, markdown report. |
| `dcsched/live.py` | Step-able simulation used by the dashboard (start/pause/reset). |
| `dcsched/policies.py` | Discovers completed checkpoints, builds a policy by name; rejects incompatible checkpoints. |
| `dcsched/plotting.py`, `palette.py` | Matplotlib figures for the report. |
| `app.py`, `dashboard/*` | Streamlit + Plotly dashboard, one module per page. |
| `scripts/train_agent.py` | CLI training. |
| `scripts/run_experiments.py` | CLI experiment pipeline. |
| `scripts/make_architecture_diagram.py` | Renders `docs/architecture.png` (needs Graphviz). |
| `tests/` | pytest suite (see TESTING.md). |
