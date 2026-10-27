"""Page 5 - recorded experiment results, history, figures and CSV export."""

import io
import json
import os
import zipfile

import pandas as pd
import streamlit as st

from . import charts
from .common import EXPERIMENTS_DIR, MODELS_DIR, DEFAULT_SIM, make_cfg
from .styles import status_badge, header_banner
from dcsched.experiments import run_experiment

FIGURES = [("thermal_summary.png", "Thermal outcomes"), ("energy_work.png", "Energy and work completed"),
           ("performance.png", "Waiting / response time and throughput"),
           ("temperature_trace.png", "Hottest-server temperature over one episode"),
           ("server_heatmaps.png", "Per-server temperature heatmaps")]


def _experiments():
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    return sorted(d for d in os.listdir(EXPERIMENTS_DIR)
                  if os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "summary.csv")))


def _zip_dir(path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def render():
    exps = _experiments()
    badge_html = status_badge("RECORDED", f"{len(exps)} RUNS") if exps else status_badge("IDLE", "NO RUNS")

    header_banner(
        "Experimental Results",
        "Empirical evaluation artifacts from results/experiments/ • Automated reports and CSV exports",
        badge_html
    )

    if exps:
        st.subheader("Experiment history")
        rows = []
        for e in exps:
            cj = os.path.join(EXPERIMENTS_DIR, e, "config.json")
            if os.path.exists(cj):
                c = json.load(open(cj))
                rows.append({"experiment": e, "created (UTC)": c.get("created_utc"),
                             "regime": c["environment_config"]["workload"]["regime"],
                             "steps": c["environment_config"]["episode_length"],
                             "eval seeds": len(c["evaluation_seeds"]),
                             "policies": len(c["policies"]),
                             "DRL runs": len(c.get("drl_runs_included", [])),
                             "runtime (s)": c.get("elapsed_seconds")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        name = st.selectbox("Open experiment", exps, index=len(exps) - 1)
        d = os.path.join(EXPERIMENTS_DIR, name)
        summary = pd.read_csv(os.path.join(d, "summary.csv"))
        raw = pd.read_csv(os.path.join(d, "raw_results.csv"))
        cfgj = json.load(open(os.path.join(d, "config.json")))

        st.subheader(f"Summary - {name}")
        cols = ["policy", "hotspot_frequency", "peak_temperature", "mean_temperature", "it_energy_kwstep",
                "cooling_energy_proxy", "mean_wait_time", "mean_response_time", "throughput",
                "completion_rate", "tasks_completed", "episode_return"]
        table = summary[["policy"]].copy()
        for m in cols[1:]:
            if f"{m}_mean" in summary:
                table[m] = summary.apply(lambda r, m=m: f"{r[m + '_mean']:.3f} ± {r[m + '_std']:.3f}", axis=1)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Mean ± s.d. over held-out seeds {cfgj['evaluation_seeds']}. "
                   "DRL rows are aggregated over their training seeds (see report for the spread).")

        metric = st.selectbox("Interactive chart metric", [m for m in cols[1:] if f"{m}_mean" in summary],
                              format_func=lambda m: m.replace("_", " "))
        st.plotly_chart(charts.metric_bars(summary, metric, metric.replace("_", " "), metric.replace("_", " "), "{:.3f}"),
                        width="stretch")

        st.subheader("Saved figures")
        for fn, cap in FIGURES:
            fp = os.path.join(d, "figures", fn)
            if os.path.exists(fp):
                st.image(fp, caption=cap, width="stretch")

        with st.expander("Automated evaluation report", expanded=True):
            st.markdown(open(os.path.join(d, "report.md"), encoding="utf-8").read())
        with st.expander("Raw per-seed results"):
            st.dataframe(raw, hide_index=True, width="stretch")
        with st.expander("Experiment configuration"):
            st.json(cfgj)

        st.subheader("Export")
        c = st.columns(5)
        for col, (fn, mime) in zip(c, [("raw_results.csv", "text/csv"), ("summary.csv", "text/csv"),
                                       ("config.json", "application/json"), ("report.md", "text/markdown")]):
            col.download_button(f"Download {fn}", open(os.path.join(d, fn), "rb").read(),
                                file_name=f"{name}_{fn}", mime=mime, key=f"dl_{fn}")
        c[4].download_button("Download everything (.zip)", _zip_dir(d), file_name=f"{name}.zip",
                             mime="application/zip", key="dl_zip")
    else:
        st.info("No recorded experiments yet. Run one below (or `python scripts/run_experiments.py`).")

    st.divider()
    st.subheader("Run a new experiment")
    st.caption("Evaluates the four baselines plus every completed DRL checkpoint on the held-out seeds, then writes "
               "CSVs, a config record, charts and a report. Only checkpoints in `results/models/` (made by "
               "`scripts/train_agent.py`) are included, not runs launched from the DRL Training page.")
    with st.form("exp_form"):
        c1, c2, c3 = st.columns(3)
        ename = c1.text_input("Experiment name", "steady-ui")
        regime = c2.selectbox("Arrival regime", ["steady", "diurnal", "bursty"])
        steps = c3.number_input("Episode length (steps)", 100, 3000, 1000, 100)
        include = st.checkbox("Include trained DRL checkpoints", True)
        go = st.form_submit_button("Run experiment", type="primary")
    if go:
        safe = "".join(ch for ch in ename.strip() if ch.isalnum() or ch in "-_")
        if not safe:
            st.error("Enter an experiment name using letters, digits, '-' or '_'.")
        else:
            cfg = make_cfg(dict(DEFAULT_SIM, regime=regime, episode_length=int(steps)))
            with st.spinner("Running experiment..."):
                run_experiment(safe, cfg, outdir_root=EXPERIMENTS_DIR, models_dir=MODELS_DIR,
                               include_drl=include, log=lambda *a: None)
            st.success(f"Experiment '{safe}' written to results/experiments/{safe}/")
            st.rerun()
