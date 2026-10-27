# Methodology

Everything here describes what the code in `dcsched/` actually does.

## 1. Problem
Place a stream of arriving tasks onto N servers (default 16, in 4 racks) so that
thermal risk, energy and task delay are jointly low. The study is a
**simulation**; no real data-centre telemetry is used.

## 2. Simulator
* **Workload** (`workload.py`): seeded Poisson arrivals. Regimes: `steady`,
  `diurnal` (sinusoidal rate) and `bursty` (base rate with transient spikes). Each task has a
  resource demand, a duration and a thermal intensity. The generator owns its
  own RNG, so every policy faces an *identical* stream for a given seed.
* **Thermal model** (`thermal.py`), lumped-parameter, per server:
  `T(t+1) = T + h·u·k + Σ_j C_ij·max(T_j − T_idle, 0) − c·(T − T_idle)`
  where `u` is utilisation, `k` the thermal intensity of hosted work, `C` a
  rack-aware recirculation (coupling) matrix and `c` the cooling rate. Config
  validation refuses parameter sets whose recirculation gain is ≥ cooling rate
  (they run away thermally).
* **Energy** (`env.py`): IT power = 0.30 (if the server hosts any task) +
  0.70·utilisation. Cooling energy is a **proxy** (proportional to how far the
  hottest server is above ambient), not a CRAC/chiller model.
* **Task lifecycle**: arrive → queue (bounded; overflow = drop) → placed on a
  server with enough capacity → runs for its duration → completes. Response
  time = wait + duration.

## 3. MDP for the DRL agent (`env.py`)
* **State** (2N+5 = 37 for N=16): per-server utilisation and normalised
  temperature, pending task (demand, duration, thermal intensity), mean
  temperature, ambient temperature.
* **Action** (N+1 discrete): choose a server, or *defer* (action N). Invalid
  (over-capacity) choices are masked in the fallback path and penalised.
* **Reward** = −[α·ThermalRisk + β·Energy + γ·Performance] with defaults
  α=1, β=0.5, γ=0.3, plus penalties for hotspots, deferring, waiting,
  dropped tasks (8.0), invalid actions (0.5) and critical temperature (60).
* A regression test guards the known reward-hacking failure
  ("always defer" scoring better than working).

## 4. Algorithms
* Baselines (`baselines.py`): round-robin, greedy-utilisation (least loaded),
  greedy-coolest (lowest temperature), random-valid.
* DRL (`agent.py`, `training.py`): Stable-Baselines3 PPO and DQN, MLP [128,128],
  resumable training with periodic checkpoints.

## 5. Fair evaluation (`evaluate.py`)
* **Held-out seeds** `EVAL_SEEDS = [1001..1005]`. Training seeds are drawn from
  a disjoint range (`FreshTraceEnv`, `check_seeds_disjoint`; enforced by tests).
* All policies are evaluated on identical traces; the arrival stream is
  independent of agent actions (tested with a poisoned-generator test).
* Metrics: hotspot rate, mean/max temperature, energy, completed tasks, dropped
  tasks, mean response time, mean queue, episode return.
* Training only ever uses the `steady` regime; evaluating on `diurnal` and
  `bursty` is a *generalisation* test, not in-distribution performance.
* Multiple training seeds are reported separately and aggregated (mean ± std);
  no best-seed cherry-picking.

## 6. Honest-reporting rules built into the pipeline
* A checkpoint is shown/used only if its progress record says `complete`.
* Charts and tables are computed from CSVs on disk; nothing is hardcoded.
* If a DRL agent loses to a baseline the report says so; `build_report`
  compares numerically rather than assuming DRL wins.
* A policy that "wins" on hotspot rate by completing no work (deferring
  forever) is flagged by also reporting completed tasks and drops.
