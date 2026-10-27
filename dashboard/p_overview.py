"""Page 1 - Overview of the live simulation session and the project's recorded evidence."""

import os
import streamlit as st

from . import charts
from .common import drl_runs, EXPERIMENTS_DIR
from .styles import status_badge, header_banner


def render():
    ss = st.session_state
    sim = ss.sim
    cfg = sim.cfg
    s = sim.summary()
    snap = sim.snapshot()
    hot = [r for r in snap if r["status"] in ("hotspot", "critical")]

    state = "finished" if sim.done else ("not started" if sim.step_count == 0 else "running")
    badge_html = status_badge(state, f"SIMULATOR: {state.upper()}")

    header_banner(
        "Thermal-Aware Workload Scheduling with Deep Reinforcement Learning",
        "Simulation-based study • IILM University, Greater Noida • B.Tech CSE, Major Project 2026-27",
        badge_html
    )

    # Cockpit Telemetry HUD Banner
    temps = [r["temperature_c"] for r in snap]
    max_t = max(temps)
    headroom = cfg.thermal.safe_temp - max_t
    headroom_color = "#34d399" if headroom > 5 else ("#fbbf24" if headroom >= 0 else "#f87171")
    mean_load = sum(r["cpu_util"] for r in snap) / len(snap)

    st.markdown(f"""
    <div class="cockpit-banner">
      <div style="display:flex; gap:20px; flex-wrap:wrap; align-items:center;">
        <div>
          <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.05em;">Active Scheduler</span>
          <div style="font-weight:700; color:#f8fafc; font-size:1.1rem;">{sim.policy.name}</div>
        </div>
        <div style="border-left:1px solid rgba(255,255,255,0.1); padding-left:16px;">
          <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.05em;">Workload Regime</span>
          <div style="font-weight:600; color:#38bdf8; font-size:1.0rem; text-transform:capitalize;">{cfg.workload.regime} (seed {sim.seed})</div>
        </div>
        <div style="border-left:1px solid rgba(255,255,255,0.1); padding-left:16px;">
          <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.05em;">Simulation Progress</span>
          <div style="font-weight:600; color:#f8fafc; font-size:1.0rem; font-family:'JetBrains Mono', monospace;">Step {sim.step_count:,} / {cfg.episode_length:,}</div>
        </div>
      </div>
      <div style="display:flex; gap:16px; align-items:center;">
        <div style="text-align:right;">
          <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.05em;">Thermal Headroom</span>
          <div style="font-weight:700; color:{headroom_color}; font-size:1.1rem; font-family:'JetBrains Mono', monospace;">{headroom:+.1f} °C</div>
        </div>
        <div style="text-align:right; border-left:1px solid rgba(255,255,255,0.1); padding-left:16px;">
          <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600; letter-spacing:0.05em;">Mean Fleet Load</span>
          <div style="font-weight:700; color:#c084fc; font-size:1.1rem; font-family:'JetBrains Mono', monospace;">{mean_load:.1%}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(sim.step_count / cfg.episode_length, 1.0))

    if sim.terminated:
        st.error("Episode terminated early: a server exceeded the critical temperature threshold.")
    elif hot:
        st.warning(f"Hotspot Alert: {len(hot)} server(s) above the safe threshold of {cfg.thermal.safe_temp:.0f} °C "
                   f"(servers {', '.join(str(r['server']) for r in hot)}).")
    else:
        st.success("Nominal: All servers operating below the safe temperature threshold.")

    # KPI Telemetry Cards (exact metric labels preserved for backend tests)
    c = st.columns(5)
    c[0].metric("Servers", f"{cfg.facility.n_servers}", f"{cfg.facility.n_racks} racks", delta_color="off")
    c[1].metric("Queue length", len(sim.env.queue), f"capacity {cfg.workload.queue_capacity}", delta_color="off")
    c[2].metric("Mean temp (°C)", f"{sum(temps) / len(temps):.1f}")
    c[3].metric("Max temp (°C)", f"{max(temps):.1f}")
    c[4].metric("Hotspot steps", f"{s['hotspot_steps']}", f"{s['hotspot_frequency']:.1%} of steps", delta_color="off")

    c = st.columns(5)
    c[0].metric("Tasks completed", s["tasks_completed"])
    c[1].metric("Tasks dropped", s["tasks_dropped"])
    c[2].metric("IT energy (kW·step)", f"{s['it_energy_kwstep']:.0f}")
    c[3].metric("Cooling proxy (kW·step)", f"{s['cooling_energy_proxy']:.0f}")
    c[4].metric("Return so far", f"{s['episode_return_so_far']:.0f}")

    # Telemetry Visualizations
    left, right = st.columns(2)
    with left:
        st.markdown("**Temperature history**")
        st.plotly_chart(charts.temperature_history(sim.history, cfg.thermal.safe_temp), width="stretch")
    with right:
        st.markdown("**Server temperatures now**")
        st.plotly_chart(charts.rack_heatmap(snap, cfg.thermal, cfg.facility.n_racks,
                                            cfg.facility.servers_per_rack), width="stretch")

    # Quick Control Bar (exact button labels preserved for tests)
    a, b, c3 = st.columns(3)
    if a.button("Advance 10 steps", disabled=sim.done):
        sim.step(10); st.rerun()
    if b.button("Run 100 steps", disabled=sim.done):
        sim.step(100); st.rerun()
    if c3.button("Run to end of episode", disabled=sim.done):
        sim.run_to_end(); st.rerun()
    st.caption("Use the **Data-Center Simulator** page to reconfigure parameters, auto-run and pause.")

    st.divider()
    st.subheader("Project evidence on disk")
    runs = drl_runs()
    exps = sorted(d for d in os.listdir(EXPERIMENTS_DIR)) if os.path.isdir(EXPERIMENTS_DIR) else []
    e1, e2, e3 = st.columns(3)
    e1.metric("Completed DRL checkpoints", len(runs))
    e2.metric("Recorded experiments", len(exps))
    e3.metric("Baseline schedulers", 4)
    if not runs:
        st.info("No completed DRL checkpoints found. Train one on the **DRL Training** page, or run "
                "`python scripts/train_agent.py` - nothing here is presented as trained until it exists.")
    if not exps:
        st.info("No recorded experiments yet. Run one on the **Experimental Results** page.")
