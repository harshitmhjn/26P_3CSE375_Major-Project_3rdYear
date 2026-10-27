"""Page 3 - compare round-robin, greedy schedulers and trained DRL schedulers."""

import pandas as pd
import streamlit as st

from . import charts
from .common import drl_runs, get_policy, make_cfg, DEFAULT_SIM, EVAL_SEEDS
from .styles import status_badge, header_banner
from dcsched.evaluate import compare_policies, headline_table
from dcsched.policies import heuristic_names

DESCRIPTIONS = {
    "round-robin": "Cycles through servers in order, skipping any without capacity. Ignores load and temperature.",
    "greedy-utilisation": "Places each task on the feasible server with the lowest CPU utilisation (conventional load balancing).",
    "greedy-coolest": "Places each task on the feasible server that is currently coolest (reactive thermal awareness) - the demanding baseline.",
    "random": "Uniformly random feasible server. A sanity reference, not a contender.",
}


def render():
    header_banner(
        "Scheduling Algorithms",
        "Rigorous benchmarking across identical held-out workload traces • Zero data leakage from training seeds",
        status_badge("BENCHMARK", "EVALUATION READY")
    )

    runs = drl_runs()
    with st.expander("What each scheduler does", expanded=False):
        for k, v in DESCRIPTIONS.items():
            st.markdown(f"* **{k}** - {v}")
        st.markdown("* **PPO / DQN** - neural-network policies trained with Stable-Baselines3 "
                    "(2 hidden layers of 128 units). They see the 2N+5 observation vector and the feasibility mask; "
                    "an infeasible greedy choice is replaced by the network's best *feasible* action.")
    if not runs:
        st.info("No completed DRL checkpoints were found, so only heuristic schedulers can be compared. "
                "Train an agent on the **DRL Training** page or with `scripts/train_agent.py`.")

    options = heuristic_names() + list(runs)
    default = [o for o in ("round-robin", "greedy-utilisation", "greedy-coolest", "PPO-seed0", "DQN-seed0") if o in options]
    selected = st.multiselect("Schedulers to compare", options, default=default)
    c1, c2, c3 = st.columns(3)
    regime = c1.selectbox("Arrival regime", ["steady", "diurnal", "bursty"])
    episode_length = c2.number_input("Episode length (steps)", 100, 3000, 1000, 100)
    n_seeds = c3.slider("Held-out seeds", 1, len(EVAL_SEEDS), len(EVAL_SEEDS))

    if st.button("Run comparison", type="primary", disabled=len(selected) < 1):
        params = dict(DEFAULT_SIM, regime=regime, episode_length=int(episode_length))
        cfg = make_cfg(params)
        try:
            policies = [get_policy(n, cfg) for n in selected]
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
            policies = None
        if policies:
            with st.spinner(f"Evaluating {len(policies)} scheduler(s) on {n_seeds} seed(s)..."):
                summary, traces, raw = compare_policies(policies, cfg, seeds=EVAL_SEEDS[:n_seeds],
                                                        collect_traces=True, return_raw=True)
            st.session_state.last_comparison = dict(
                summary=summary, traces=traces, raw=raw, cfg=cfg,
                label=f"{regime} arrivals, {int(episode_length)} steps, seeds {EVAL_SEEDS[:n_seeds]}")

    res = st.session_state.last_comparison
    if res is None:
        st.info("Select schedulers and press **Run comparison**.")
        return

    summary, traces, raw, cfg = res["summary"], res["traces"], res["raw"], res["cfg"]

    st.subheader("Results")
    st.caption(res["label"] + " - mean ± s.d. across seeds")

    # High-level benchmark summary chips
    best_thermal = summary.loc[summary["hotspot_frequency_mean"].idxmin()]
    best_energy = summary.loc[summary["it_energy_kwstep_mean"].idxmin()]
    best_comp = summary.loc[summary["tasks_completed_mean"].idxmax()]

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Lowest Hotspot Frequency", f"{best_thermal['hotspot_frequency_mean']:.1%}", f"{best_thermal['policy']}")
    sc2.metric("Lowest IT Energy", f"{best_energy['it_energy_kwstep_mean']:.0f} kW·step", f"{best_energy['policy']}")
    sc3.metric("Highest Completed Tasks", f"{best_comp['tasks_completed_mean']:.0f}", f"{best_comp['policy']}")

    st.markdown("**Headline Comparison Metrics**")
    st.dataframe(headline_table(summary), hide_index=True, width="stretch")

    st.markdown("**Multi-Criteria Tradeoff Analysis**")
    st.plotly_chart(charts.radar_comparison(summary), width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(charts.metric_bars(summary, "hotspot_frequency", "Hotspot frequency (lower is better)",
                                      "Fraction of steps above safe temp"), width="stretch")
    b.plotly_chart(charts.metric_bars(summary, "peak_temperature", "Peak server temperature (lower is better)",
                                      "°C", "{:.1f}"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(charts.metric_bars(summary, "it_energy_kwstep", "IT energy (kW·step)", "kW·step", "{:.0f}"),
                   width="stretch")
    b.plotly_chart(charts.metric_bars(summary, "mean_wait_time", "Mean waiting time (steps)", "steps", "{:.2f}"),
                   width="stretch")
    st.plotly_chart(charts.trace_comparison(traces, cfg.thermal.safe_temp,
                                            title=f"Hottest-server temperature, held-out seed {EVAL_SEEDS[0]}"),
                    width="stretch")

    done_diff = summary["tasks_completed_mean"].max() - summary["tasks_completed_mean"].min()
    if done_diff > 0.02 * summary["tasks_completed_mean"].max():
        st.warning("Schedulers completed noticeably different amounts of work. A policy that looks thermally better "
                   "may simply be doing less work (for example by deferring) - compare the *Completed* column.")
    with st.expander("Per-seed raw results"):
        st.dataframe(raw, hide_index=True, width="stretch")
    st.download_button("Download this comparison (CSV)", raw.to_csv(index=False).encode(),
                       file_name="comparison_raw_results.csv", mime="text/csv")
