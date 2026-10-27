"""Page 2 - interactive data-centre simulator."""

import pandas as pd
import streamlit as st

from . import charts
from .common import (drl_runs, get_policy, rebuild_simulation, STATUS_ICON, DEFAULT_SIM)
from .styles import status_badge, header_banner
from dcsched.policies import heuristic_names


def _on_policy_change():
    ss = st.session_state
    try:
        ss.sim.set_policy(get_policy(ss.live_policy, ss.sim.cfg))
        ss.sim_params["policy"] = ss.live_policy
        ss.policy_error = None
    except (ValueError, FileNotFoundError) as exc:
        ss.policy_error = str(exc)


def _config_form(p):
    with st.expander("Configuration (applies on reset)", expanded=False):
        with st.form("sim_config"):
            c1, c2, c3, c4 = st.columns(4)
            n_servers = c1.selectbox("Servers", [8, 12, 16, 24, 32], index=[8, 12, 16, 24, 32].index(p["n_servers"]))
            n_racks = c2.selectbox("Racks", [1, 2, 3, 4, 6, 8], index=[1, 2, 3, 4, 6, 8].index(p["n_racks"]))
            regime = c3.selectbox("Arrival regime", ["steady", "diurnal", "bursty"],
                                  index=["steady", "diurnal", "bursty"].index(p["regime"]))
            seed = c4.number_input("Workload seed", 0, 10_000_000, int(p["seed"]))
            c1, c2, c3, c4 = st.columns(4)
            arrival_rate = c1.slider("Mean arrivals / step", 0.1, 2.0, float(p["arrival_rate"]), 0.05)
            episode_length = c2.number_input("Episode length (steps)", 50, 5000, int(p["episode_length"]), 50)
            queue_capacity = c3.number_input("Queue capacity", 1, 100, int(p["queue_capacity"]))
            options = heuristic_names() + list(drl_runs())
            policy = c4.selectbox("Scheduler", options,
                                  index=options.index(p["policy"]) if p["policy"] in options else 0)
            with st.expander("Advanced: thermal and reward parameters"):
                a1, a2, a3, a4, a5 = st.columns(5)
                heat = a1.number_input("heat_per_util (°C/step)", 0.5, 20.0, float(p["heat_per_util"]), 0.5)
                cool = a2.number_input("cooling_rate", 0.01, 1.0, float(p["cooling_rate"]), 0.01)
                safe = a3.number_input("safe_temp (°C)", 35.0, 74.0, float(p["safe_temp"]), 1.0)
                alpha = a4.number_input("α thermal weight", 0.0, 10.0, float(p["alpha"]), 0.1)
                beta = a5.number_input("β energy weight", 0.0, 10.0, float(p["beta"]), 0.1)
                gamma_w = st.number_input("γ performance weight", 0.0, 10.0, float(p["gamma_w"]), 0.1)
            go = st.form_submit_button("Apply configuration & reset")
        if go:
            new = dict(n_servers=n_servers, n_racks=n_racks, regime=regime, seed=int(seed),
                       arrival_rate=arrival_rate, episode_length=int(episode_length),
                       queue_capacity=int(queue_capacity), policy=policy, heat_per_util=heat,
                       cooling_rate=cool, safe_temp=safe, alpha=alpha, beta=beta, gamma_w=gamma_w)
            err = rebuild_simulation(new)
            st.session_state.auto_run = False
            if err:
                st.error(f"Configuration rejected: {err}")
            else:
                st.success("Configuration applied; simulation reset.")
                st.rerun()


def _live_panel():
    ss = st.session_state
    sim = ss.sim
    cfg = sim.cfg
    if ss.auto_run and not sim.done:
        sim.step(ss.tick_steps)
        if sim.done:
            st.rerun()                         # leave auto-run mode cleanly
    snap = sim.snapshot()
    s = sim.summary()

    # Progress and status
    st.progress(min(sim.step_count / cfg.episode_length, 1.0),
                text=f"Step {sim.step_count:,} / {cfg.episode_length:,} • Active Policy: {sim.policy.name}"
                     + (" • EPISODE COMPLETED" if sim.done else ""))

    if sim.terminated:
        st.error("Terminated: a server exceeded the critical temperature threshold.")

    # Status breakdown pills
    n_normal = sum(1 for r in snap if r["status"] == "normal")
    n_warm = sum(1 for r in snap if r["status"] == "warm")
    n_hot = sum(1 for r in snap if r["status"] == "hotspot")
    n_crit = sum(1 for r in snap if r["status"] == "critical")

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:600;">Server Status:</span>
        <span class="hud-pill hud-pill-green"><span class="pulse-dot pulse-green"></span>{n_normal} Nominal</span>
        <span class="hud-pill hud-pill-amber"><span class="pulse-dot pulse-amber"></span>{n_warm} Warm</span>
        <span class="hud-pill hud-pill-orange"><span class="pulse-dot pulse-red"></span>{n_hot} Hotspot</span>
        {f'<span class="hud-pill hud-pill-red"><span class="pulse-dot pulse-red"></span>{n_crit} Critical</span>' if n_crit else ''}
      </div>
      <div style="font-size:0.8rem; color:#94a3b8;">
        Thermal Safe Limit: <strong style="color:#ef4444;">{cfg.thermal.safe_temp:.0f}°C</strong> | Ambient: <strong style="color:#38bdf8;">{cfg.thermal.ambient_temp:.0f}°C</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    k = st.columns(6)
    k[0].metric("Max temp (°C)", f"{max(r['temperature_c'] for r in snap):.1f}")
    k[1].metric("Hotspot steps", s["hotspot_steps"])
    k[2].metric("Queue", len(sim.env.queue))
    k[3].metric("Running tasks", sum(r["n_tasks"] for r in snap))
    k[4].metric("Completed", s["tasks_completed"])
    k[5].metric("Dropped", s["tasks_dropped"])

    st.plotly_chart(charts.rack_heatmap(snap, cfg.thermal, cfg.facility.n_racks,
                                        cfg.facility.servers_per_rack), width="stretch")
    t1, t2, t3, t4, t5 = st.tabs(["Temperature", "Queue & throughput", "Power", "Per-server", "Tables"])
    with t1:
        st.plotly_chart(charts.temperature_history(sim.history, cfg.thermal.safe_temp), width="stretch")
    with t2:
        st.plotly_chart(charts.queue_and_completed(sim.history), width="stretch")
    with t3:
        st.plotly_chart(charts.power_history(sim.history), width="stretch")
        st.caption("Cooling power is a proxy (coefficient × hottest-server excess over ambient), not a facility cooling model.")
    with t4:
        st.plotly_chart(charts.server_bars(snap, cfg.thermal.safe_temp), width="stretch")
    with t5:
        df = pd.DataFrame(snap)
        df["status"] = df["status"].map(lambda x: f"{STATUS_ICON.get(x, '•')} {x.upper()}")
        df["task_ids"] = df["task_ids"].map(lambda ids: ", ".join(map(str, ids)) if ids else "none")
        df = df.rename(columns={"cpu_util": "CPU util", "temperature_c": "temp °C", "power_kw": "power kW",
                                "n_tasks": "tasks", "task_ids": "assigned task IDs"})
        st.markdown("**Servers and their workload assignments**")
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"CPU util": st.column_config.ProgressColumn("CPU util", min_value=0, max_value=1, format="%.0f%%"),
                                    "temp °C": st.column_config.NumberColumn(format="%.1f"),
                                    "power kW": st.column_config.NumberColumn(format="%.2f")})
        st.markdown("**Pending queue** (head of queue is placed next)")
        q = sim.queue_view()
        st.dataframe(pd.DataFrame(q) if q else pd.DataFrame({"queue": ["empty"]}), hide_index=True, width="stretch")


def render():
    ss = st.session_state
    sim = ss.sim

    status_str = "FINISHED" if sim.done else ("AUTO-RUNNING" if ss.auto_run else "PAUSED / READY")
    badge_html = status_badge(status_str)

    header_banner(
        "Data-Center Simulator",
        "Interactive digital twin environment • Step, inspect and evaluate live thermal and workload dynamics",
        badge_html
    )

    _config_form(ss.sim_params)

    options = heuristic_names() + list(drl_runs())
    cur = sim.policy.name if sim.policy.name in options else options[0]
    c = st.columns([2, 1, 1, 1, 1, 1])
    c[0].selectbox("Scheduler (switch keeps state)", options, index=options.index(cur),
                   key="live_policy", on_change=_on_policy_change)
    if ss.get("policy_error"):
        st.error(ss.policy_error)
    acted = False
    if c[1].button("Step ×1", disabled=sim.done):
        sim.step(1); acted = True
    if c[2].button("Step ×10", disabled=sim.done):
        sim.step(10); acted = True
    if c[3].button("Run ×100", disabled=sim.done):
        sim.step(100); acted = True
    if c[4].button("Run to end", disabled=sim.done):
        sim.run_to_end(); acted = True
    if c[5].button("Reset", type="primary"):
        sim.reset(); ss.auto_run = False; acted = True
    if acted:
        st.rerun()          # re-render so button enabled/disabled state matches the new simulation state

    p = st.columns([1, 1, 3])
    p[0].toggle("Auto-run (turn off to pause)", key="auto_run", disabled=sim.done)
    p[1].slider("Steps per tick", 1, 25, key="tick_steps")

    interval = 0.7 if (ss.auto_run and not sim.done) else None
    st.fragment(run_every=interval)(_live_panel)()
