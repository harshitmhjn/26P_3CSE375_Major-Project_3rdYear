"""Page 4 - DRL training: status, curves, hyperparameters, checkpoints, evaluation; launch training."""

import json
import os

import pandas as pd
import streamlit as st

from . import charts
from .common import (MODELS_DIR, UI_MODELS_DIR, EVAL_SEEDS, launch_training, tail, ROOT)
from .styles import status_badge, header_banner
from dcsched.agent import default_hyperparameters
from dcsched.policies import iter_model_dirs
from dcsched.training import list_runs, load_curve


def _collect_runs():
    rows = []
    for base in (MODELS_DIR, UI_MODELS_DIR):
        for d in iter_model_dirs(base):
            if d == UI_MODELS_DIR:
                continue
            for rec in list_runs(d):
                rec["run_dir"] = d
                rows.append(rec)
    return rows


def render():
    ss = st.session_state

    tp = ss.train_proc
    training_active = tp and tp["proc"].poll() is None
    badge_status = "RUNNING" if training_active else "IDLE"
    badge_html = status_badge(badge_status, f"TRAINING: {badge_status}")

    header_banner(
        "DRL Training",
        "Stable-Baselines3 PPO & DQN training pipeline • Checkpoints, learning curves, and hyperparameter telemetry",
        badge_html
    )

    # Neural Policy Architecture Visual Card
    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(22,30,49,0.7) 0%, rgba(13,19,33,0.85) 100%);
                border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px 20px; margin-bottom:16px; backdrop-filter:blur(10px);">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
          <span style="font-size:0.75rem; text-transform:uppercase; color:#f97316; font-weight:700; letter-spacing:0.05em;">Neural Architecture</span>
          <div style="color:#f8fafc; font-weight:600; font-size:1.05rem; margin-top:2px;">Multi-Layer Perceptron (Actor-Critic / Deep Q-Network)</div>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
          <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 12px; text-align:center;">
            <div style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase;">Input Space</div>
            <div style="color:#38bdf8; font-weight:700; font-family:'JetBrains Mono', monospace;">2N + 5</div>
          </div>
          <span style="color:#64748b;">➔</span>
          <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 12px; text-align:center;">
            <div style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase;">Hidden Layers</div>
            <div style="color:#c084fc; font-weight:700; font-family:'JetBrains Mono', monospace;">[128, 128] ReLU</div>
          </div>
          <span style="color:#64748b;">➔</span>
          <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 12px; text-align:center;">
            <div style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase;">Action Head</div>
            <div style="color:#34d399; font-weight:700; font-family:'JetBrains Mono', monospace;">N + 1 (Masked)</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    runs = _collect_runs()
    st.subheader("Runs and checkpoints")
    if not runs:
        st.info("No training runs found yet. Start one below or run `python scripts/train_agent.py --algo PPO`.")
    else:
        rows = []
        for r in runs:
            ev = r.get("eval", {})
            rows.append({
                "run": f"{r['algo']}-seed{r['seed']}",
                "folder": os.path.relpath(r["run_dir"], ROOT),
                "status": r["status"],
                "timesteps": f"{r['timesteps_done']:,} / {r['timesteps_target']:,}",
                "train time (s)": round(r["train_seconds"]),
                "episodes": r.get("n_training_episodes", 0),
                "checkpoint": "yes" if r["has_checkpoint"] else "no",
                "held-out hotspot freq.": round(ev["hotspot_frequency_mean"], 3) if ev else None,
                "held-out completed": round(ev["tasks_completed_mean"], 1) if ev else None,
                "held-out return": round(ev["episode_return_mean"]) if ev else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Held-out columns are means over evaluation seeds {EVAL_SEEDS}; training used different seeds.")

        st.subheader("Learning curves")
        labels = {f"{r['algo']}-seed{r['seed']} ({os.path.basename(r['run_dir'])})": r for r in runs}
        chosen = st.multiselect("Runs to plot", list(labels), default=list(labels)[:6])
        window = st.slider("Smoothing window (episodes)", 1, 50, 10)
        curves = {}
        for lab in chosen:
            r = labels[lab]
            c = load_curve(r["run_dir"], r["algo"], r["seed"])
            if c is not None and len(c["returns"]):
                curves[f"{r['algo']}-seed{r['seed']}"] = c
        if curves:
            st.plotly_chart(charts.learning_curves(curves, window), width="stretch")
            st.caption("One training episode = one fresh workload trace of the configured length. "
                       "A rising curve means the agent is earning higher (less negative) reward.")
        else:
            st.info("No learning-curve data for the selected runs yet.")

    st.subheader("Hyperparameters")
    h = st.columns(2)
    for col, algo in zip(h, ("PPO", "DQN")):
        hp = default_hyperparameters(algo)
        col.markdown(f"**{algo}** (defaults used by `scripts/train_agent.py`)")
        col.dataframe(pd.DataFrame({"parameter": list(hp), "value": [str(v) for v in hp.values()]}),
                      hide_index=True, width="stretch")

    st.subheader("Launch a training run")
    st.caption("Starts `scripts/train_agent.py` as a background process and writes to its own folder under "
               "`results/models_ui/`. For the full campaign (300k timesteps x 3 seeds) use the command line.")
    with st.form("train_form"):
        c1, c2, c3, c4 = st.columns(4)
        algo = c1.selectbox("Algorithm", ["PPO", "DQN"])
        steps = c2.number_input("Timesteps", 2000, 2_000_000, 20_000, 1000)
        seed = c3.number_input("Training seed", 0, 1_000_000, 10)
        regime = c4.selectbox("Arrival regime", ["steady", "diurnal", "bursty"])
        w1, w2, w3 = st.columns(3)
        alpha = w1.number_input("α thermal", 0.0, 10.0, 1.0, 0.1)
        beta = w2.number_input("β energy", 0.0, 10.0, 0.5, 0.1)
        gamma_w = w3.number_input("γ performance", 0.0, 10.0, 0.3, 0.1)
        go = st.form_submit_button("Start training", type="primary")
    if go:
        running = ss.train_proc and ss.train_proc["proc"].poll() is None
        if running:
            st.warning("A training run launched from this dashboard is still running.")
        else:
            try:
                ss.train_proc = launch_training(algo, int(steps), int(seed), regime, alpha, beta, gamma_w)
                st.success(f"Started {ss.train_proc['run_id']}")
            except ValueError as exc:
                st.error(str(exc))

    tp = ss.train_proc
    if tp:
        code = tp["proc"].poll()
        state = "running" if code is None else ("finished" if code == 0 else f"failed (exit code {code})")
        st.markdown(f"**Dashboard-launched run `{tp['run_id']}`: {state}**")
        prog = os.path.join(tp["outdir"], f"{tp['algo'].lower()}_seed{tp['seed']}_progress.json")
        if os.path.exists(prog):
            rec = json.load(open(prog))
            st.progress(min(rec["timesteps_done"] / rec["timesteps_target"], 1.0),
                        text=f"{rec['timesteps_done']:,} / {rec['timesteps_target']:,} timesteps ({rec['status']})")
        st.code(tail(os.path.join(tp["outdir"], "train.log")) or "(no output yet)", language="text")
        if st.button("Refresh status"):
            st.rerun()
