"""Page 6 - architecture, methodology, assumptions and limitations.

Parameter values are read from the configuration dataclasses so this page cannot
drift from the code. The implementation-status block is derived from files that
actually exist on disk."""

import os
from dataclasses import asdict

import pandas as pd
import streamlit as st

from .common import drl_runs, EXPERIMENTS_DIR, ROOT
from .styles import status_badge, header_banner
from dcsched import EnvConfig
from dcsched.agent import default_hyperparameters
from dcsched.evaluate import EVAL_SEEDS

DOT = """
digraph G {
  bgcolor="transparent";
  rankdir=LR; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fillcolor="#111827", color="#374151", fontcolor="#f8fafc", fontname="Inter, Helvetica", fontsize=10];
  edge [color="#64748b", fontcolor="#94a3b8", fontname="Inter, Helvetica", fontsize=9];

  cfg   [label="config.py\\nEnvConfig (validated,\\nJSON round-trip)", fillcolor="#1e293b", color="#475569"];
  wl    [label="workload.py\\nseeded Poisson tasks\\nsteady | diurnal | bursty", fillcolor="#1e293b", color="#475569"];
  th    [label="thermal.py\\nlumped-parameter model\\n+ stability check", fillcolor="#1e293b", color="#475569"];
  env   [label="env.py\\nDataCenterSchedulingEnv\\n(Gymnasium)", fillcolor="#2a1711", color="#f97316", fontcolor="#ffedd5", penwidth=2.0];
  base  [label="baselines.py\\nround-robin | greedy-util\\ngreedy-coolest | random", fillcolor="#1e293b", color="#475569"];
  agent [label="agent.py\\nPPO / DQN (SB3)\\n+ PolicyAgent wrapper", fillcolor="#2b1424", color="#ec4899", fontcolor="#fdf2f8", penwidth=1.8];
  train [label="training.py\\nresumable, fresh traces,\\ncheckpoints", fillcolor="#1e293b", color="#475569"];
  ev    [label="evaluate.py\\nheld-out seeds\\nmetrics", fillcolor="#1e293b", color="#475569"];
  exp   [label="experiments.py\\nCSV | report.md | figures", fillcolor="#1e293b", color="#475569"];
  ui    [label="app.py + dashboard/\\nStreamlit + Plotly", fillcolor="#0c2340", color="#38bdf8", fontcolor="#e0f2fe", penwidth=2.0];

  cfg -> env; wl -> env [label="task stream"]; th -> env [label="temperatures"];
  env -> train [label="obs, reward"]; train -> agent [label="learns"];
  env -> ev; base -> ev [label="act(env)"]; agent -> ev [label="act(env)"];
  ev -> exp; exp -> ui [label="results/"]; agent -> ui; base -> ui; env -> ui [label="live.py"];
}
"""


def _status_rows():
    runs = drl_runs()
    exps = [d for d in (os.listdir(EXPERIMENTS_DIR) if os.path.isdir(EXPERIMENTS_DIR) else [])
            if os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "summary.csv"))]
    has_tests = os.path.isdir(os.path.join(ROOT, "tests"))
    algos = sorted({r["algo"] for r in runs.values()})
    return [
        ("Simulator, thermal & energy models, workload generator", "implemented"),
        ("Baseline schedulers (round-robin, greedy-utilisation, greedy-coolest, random)", "implemented"),
        ("DRL environment, PPO/DQN training pipeline, checkpoints", "implemented"),
        ("Trained DRL checkpoints", f"{len(runs)} completed ({', '.join(algos)})" if runs else "none found"),
        ("Recorded experiments on held-out seeds", f"{len(exps)} found" if exps else "none found"),
        ("Automated tests", "present (run `pytest`)" if has_tests else "missing"),
        ("Reward-weight sensitivity study", "supported by `--alpha/--beta/--gamma-w`; run results are not shown here unless recorded"),
        ("Validation against a physical data centre", "NOT done - out of scope"),
    ]


def render():
    cfg = EnvConfig()
    n = cfg.facility.n_servers

    header_banner(
        "Architecture and Methodology",
        "Formal Markov Decision Process, lumped-parameter heat physics, and verification harness",
        status_badge("SPECIFICATION", "RIGOROUS DESIGN")
    )

    st.subheader("System architecture")
    st.graphviz_chart(DOT)
    st.markdown(
        "The simulator (`env.py`) is the single source of truth: the dashboard, the training pipeline and the "
        "experiment pipeline all drive the *same* environment through the same `act(env)` policy interface, so "
        "no scheduler gets a different view of the data centre. UI code never contains simulation logic.")

    st.subheader("Problem formulation (Markov Decision Process)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div style="background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-left:3px solid #38bdf8; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
<div style="font-weight:700; color:#38bdf8; margin-bottom:6px;">STATE SPACE (Continuous)</div>
Vector dimension <b>2N + 5 (= {2 * n + 5} for N = {n} servers)</b>:
<ul>
  <li>Per-server CPU utilisation (N)</li>
  <li>Per-server temperature, normalised by {cfg.thermal.temp_norm:.0f} °C (N)</li>
  <li>Pending task: CPU demand, duration, thermal intensity (3)</li>
  <li>Mean facility temperature and ambient temperature (2)</li>
</ul>
<div style="font-weight:700; color:#38bdf8; margin:10px 0 6px 0;">ACTION SPACE (Discrete)</div>
<b>N + 1 = {n + 1}</b> discrete actions: place task on server 0 … N-1, or <b>defer</b>.
<br><small style="color:#94a3b8;">Placements exceeding server capacity are <i>masked out</i>. Selecting a masked action defaults to deferral with invalid_penalty = {cfg.reward.invalid_penalty}.</small>
</div>
""", unsafe_allow_html=True)

    with c2:
        r = cfg.reward
        st.markdown(f"""
<div style="background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-left:3px solid #f97316; border-radius:10px; padding:14px 18px; margin-bottom:12px;">
<div style="font-weight:700; color:#f97316; margin-bottom:6px;">REWARD FORMULATION</div>
<b>Reward (per step) = −[ α·ThermalRisk + β·Energy + γ·Performance ]</b>
<br><small style="color:#cbd5e1;">Configured weights: α = {r.alpha}, β = {r.beta}, γ = {r.gamma_w}</small>
<ul>
  <li><b>ThermalRisk:</b> Σ (T − T_safe)⁺² / 10 + fixed {r.hotspot_penalty} when any T > {cfg.thermal.safe_temp:.0f} °C</li>
  <li><b>Energy:</b> (IT power + cooling proxy) / {r.energy_scale:.0f}</li>
  <li><b>Performance:</b> {r.wait_penalty}·queue + {r.defer_penalty}·defer + {r.drop_penalty}·dropped + invalid penalty</li>
  <li><b>Terminal Critical:</b> extra −{r.critical_penalty:.0f} penalty if any T > {cfg.thermal.critical_temp:.0f} °C</li>
</ul>
<small style="color:#94a3b8;"><b>Drop Penalty Defense:</b> Prevents degenerative exploit where agent defers until queue overflows to avoid thermal penalties.</small>
</div>
""", unsafe_allow_html=True)

    st.subheader("Thermal model")
    st.latex(r"T_i(t{+}1)=T_i(t)+h\,u_i\,k_i+\sum_{j\neq i}C_{ij}\,\max(T_j-T_{idle},0)-c\,(T_i-T_{idle})")
    th = asdict(cfg.thermal)
    units = {"ambient_temp": "°C", "idle_temp": "°C", "heat_per_util": "°C / step", "recirc_same_rack": "fraction / step",
             "recirc_cross_rack": "fraction / step", "cooling_rate": "1 / step", "safe_temp": "°C",
             "critical_temp": "°C", "temp_norm": "°C"}
    st.dataframe(pd.DataFrame([{"parameter": k, "default": v, "unit": units.get(k, "")} for k, v in th.items()]),
                 hide_index=True, width="stretch")
    st.markdown("`h` = heat_per_util, `k_i` = demand-weighted thermal intensity of server *i*'s tasks, `c` = cooling_rate, "
                "`C_ij` = recirculation (stronger within a rack, decaying with distance). The idle state is an exact fixed "
                "point, and the constructor **refuses** configurations where recirculation gain ≥ cooling rate "
                "(they would diverge whatever the scheduler does).")

    st.subheader("Energy model")
    st.markdown(f"""
* **IT power** per server (kW) = {cfg.reward.idle_power} (idle, if hosting any task) + {cfg.reward.dynamic_power} × utilisation. Energy = Σ power × 1 step (kW·step).
* **Cooling** is a *proxy*: `{cfg.reward.cooling_coeff}` × (hottest server − ambient) kW. It is **not** a facility cooling model, so
  no claim of cooling *savings* in kWh should be drawn from it - it only rewards keeping the hottest server cool.
""")

    st.subheader("Workload model and assumptions")
    w = cfg.workload
    st.markdown(f"""
* Poisson arrivals, mean {w.arrival_rate}/step; regimes: **steady**, **diurnal** (sinusoidal, ±{w.diurnal_amplitude:.0%}), **bursty** (×{w.burst_multiplier:g} bursts).
* Task: CPU demand U[{w.demand_min}, {w.demand_max}], duration U[{w.duration_min}, {w.duration_max}] steps, thermal intensity U[{w.thermal_min}, {w.thermal_max}]
  (≥ 1.0 ⇒ *compute-bound*, else *I/O-bound*). Queue capacity {w.queue_capacity}; overflow tasks are dropped.
* **Only CPU is modelled** (no memory). Demand is constant for a task's lifetime; no preemption or migration.
* Lifecycle: arrived → queued → placed/running → completed (or dropped at the queue).
""")

    st.subheader("Evaluation procedure")
    st.markdown(f"""
* Every policy is run through the same loop on **identical held-out workload traces** (seeds {EVAL_SEEDS}); the generator owns its RNG, so the task stream does not depend on the policy.
* Training uses a fresh trace each episode drawn from a stream that can never produce an evaluation seed; training seeds that overlap evaluation seeds are refused.
* Metrics (per episode): hotspot frequency = steps with hottest server > T_safe ÷ steps; peak / mean / mean-max temperature; mean spatial variance;
  IT energy and cooling proxy (kW·step); mean wait (placement − arrival) and response (finish − arrival = wait + duration);
  throughput = completed ÷ steps; completion rate = completed ÷ arrived; mean utilisation; load imbalance = mean over steps of the s.d. of utilisation across servers.
  Aggregation: mean and population s.d. across evaluation seeds; DRL rows are first averaged over training seeds.
* Policies see only public state (utilisation, temperatures, pending task, feasibility mask) - never future arrivals (enforced by a test).
""")
    hp = {a: default_hyperparameters(a) for a in ("PPO", "DQN")}
    st.markdown(f"DRL agents: MLP policy {hp['PPO']['net_arch']}, discount γ = {hp['PPO']['gamma']}, "
                f"PPO lr {hp['PPO']['learning_rate']}, DQN lr {hp['DQN']['learning_rate']} "
                "(full tables on the DRL Training page).")

    st.subheader("Limitations")
    st.markdown("""
* The thermal model is a **simplified lumped-parameter approximation**; it is not CFD and has **not** been validated against a real facility. Absolute temperatures are illustrative.
* The observation does not include queue length or temperature history, so the problem is only approximately Markovian.
* PPO/DQN are trained **without action masking** (mask is applied only at evaluation); `sb3-contrib` MaskablePPO is a natural future improvement.
* Results come from one facility size (16 servers / 4 racks); checkpoints do not transfer to other sizes.
* Workload is synthetic; no real traces. Single-resource (CPU) model; no migration.
* Training budgets are modest (see DRL Training page); more training or tuning could change conclusions.
""")

    st.subheader("Implementation status (derived from files on disk)")
    st.dataframe(pd.DataFrame(_status_rows(), columns=["Component", "Status"]), hide_index=True, width="stretch")
