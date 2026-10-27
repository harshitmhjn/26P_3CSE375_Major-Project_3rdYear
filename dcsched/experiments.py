"""Reproducible experiment pipeline (Phase-II report, Sections 2.4-2.5).

``run_experiment`` evaluates a list of policies on identical held-out workload
traces and writes, under ``results/experiments/<name>/``:

  raw_results.csv     one row per (policy, seed) with every metric
  summary.csv         per-policy mean and std across seeds
  config.json         full environment configuration, seeds, model files used
  report.md           concise automated evaluation report (honest ranking,
                      including cases where the DRL agent does not win)
  figures/*.png       thermal, energy/work, performance, temperature traces,
                      per-server heatmaps

Nothing in this module hardcodes a result: every number in the report and every
bar in every chart is computed from the raw rows.
"""

import json
import os
import platform
import time

import numpy as np
import pandas as pd

from .config import EnvConfig
from .evaluate import EVAL_SEEDS, compare_policies, METRIC_ORDER
from .baselines import RandomValid, RoundRobin, GreedyUtilisation, GreedyCoolest
from . import plotting
from .palette import base_policy, sort_policies

BASELINE_POLICIES = lambda: [RandomValid(seed=0), RoundRobin(), GreedyUtilisation(), GreedyCoolest()]

# Metrics where a LOWER value is better (used when ranking in the report).
LOWER_IS_BETTER = {
    "hotspot_frequency", "peak_temperature", "mean_temperature", "mean_max_temperature",
    "mean_spatial_variance", "total_energy", "it_energy_kwstep", "cooling_energy_proxy",
    "mean_wait_time", "mean_response_time", "mean_queue_length", "load_imbalance",
    "tasks_dropped", "defer_actions", "invalid_actions", "critical_terminations",
}


def discover_agents(models_dir: str):
    """Load every trained checkpoint found in ``models_dir``.

    Returns a list of (algo, seed, PolicyAgent). Runs whose training did not
    complete are skipped (their status is reported by the caller), so a partial
    model is never presented as a trained result.
    """
    from .agent import load_model, PolicyAgent
    from .training import list_runs

    agents = []
    for rec in list_runs(models_dir):
        if rec.get("status") != "complete" or not rec.get("has_checkpoint"):
            continue
        model = load_model(rec["algo"], rec["model_path"])
        agents.append((rec["algo"], rec["seed"],
                       PolicyAgent(model, name=f"{rec['algo']}-seed{rec['seed']}")))
    return agents


def aggregate_drl(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-training-seed DRL rows into one row per algorithm.

    For each (algo, eval seed) the metric is averaged over training seeds; the
    reported std is then taken over evaluation seeds, matching the baselines.
    Also returns the spread across training seeds in ``*_trainseed_std``.
    """
    rows = []
    for algo in ("PPO", "DQN"):
        sub = raw[raw["policy"].str.startswith(f"{algo}-seed")]
        if sub.empty:
            continue
        n_train = sub["policy"].nunique()
        agg = {"policy": algo, "n_training_seeds": n_train}
        per_policy = sub.groupby("policy")
        for m in METRIC_ORDER:
            if m not in sub.columns:
                continue
            by_eval_seed = sub.groupby("seed")[m].mean()
            agg[f"{m}_mean"] = float(by_eval_seed.mean())
            agg[f"{m}_std"] = float(by_eval_seed.std(ddof=0))
            agg[f"{m}_trainseed_std"] = float(per_policy[m].mean().std(ddof=0)) if n_train > 1 else 0.0
        rows.append(agg)
    return pd.DataFrame(rows)


def run_experiment(name: str, cfg: EnvConfig, outdir_root: str = "results/experiments",
                   models_dir: str = "results/models", seeds=None, include_drl: bool = True,
                   make_figures: bool = True, log=print):
    seeds = list(seeds or EVAL_SEEDS)
    outdir = os.path.join(outdir_root, name)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    policies = BASELINE_POLICIES()
    drl_files = []
    if include_drl and os.path.isdir(models_dir):
        try:
            for algo, seed, agent in discover_agents(models_dir):
                policies.append(agent)
                drl_files.append(f"{algo}-seed{seed}")
        except ImportError as exc:
            log(f"  [skip DRL] {exc}")

    log(f"Experiment '{name}': {len(policies)} policies x {len(seeds)} held-out seeds "
        f"({cfg.workload.regime} arrivals, {cfg.episode_length} steps)")
    t0 = time.time()
    summary, traces, raw = compare_policies(policies, cfg, seeds=seeds,
                                            collect_traces=True, return_raw=True)
    elapsed = time.time() - t0

    baseline_summary = summary[~summary["policy"].str.contains("-seed")]
    drl_agg = aggregate_drl(raw)
    # Headline table: baselines plus one aggregated row per DRL algorithm.
    headline = pd.concat([baseline_summary, drl_agg], ignore_index=True, sort=False)

    raw.to_csv(os.path.join(outdir, "raw_results.csv"), index=False)
    headline.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "summary_per_run.csv"), index=False)

    config_record = {
        "experiment": name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_seeds": seeds,
        "policies": list(summary["policy"]),
        "drl_runs_included": drl_files,
        "elapsed_seconds": round(elapsed, 2),
        "python": platform.python_version(),
        "environment_config": cfg.to_dict(),
    }
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(config_record, fh, indent=2)

    figures = {}
    if make_figures:
        sub = f"{cfg.workload.regime} arrivals, mean +/- s.d. over {len(seeds)} held-out seeds"
        # Figures use the aggregated headline rows (one bar per algorithm).
        figures["thermal"] = plotting.fig_thermal_summary(headline, os.path.join(figdir, "thermal_summary.png"), sub)
        figures["energy"] = plotting.fig_energy_work(headline, os.path.join(figdir, "energy_work.png"), sub)
        figures["performance"] = plotting.fig_performance(headline, os.path.join(figdir, "performance.png"), sub)
        # Trace panels: baselines plus, if present, the first run of each algorithm.
        trace_sel = {k: v for k, v in traces.items() if "-seed" not in k}
        for algo in ("PPO", "DQN"):
            first = [k for k in sorted(traces) if k.startswith(f"{algo}-seed")]
            if first:
                trace_sel[algo] = traces[first[0]]
        figures["trace"] = plotting.fig_temperature_traces(
            trace_sel, cfg.thermal.safe_temp, os.path.join(figdir, "temperature_trace.png"),
            seed_label=f"held-out seed {seeds[0]}")
        figures["heatmap"] = plotting.fig_server_heatmaps(
            trace_sel, cfg.facility, cfg.thermal.safe_temp, os.path.join(figdir, "server_heatmaps.png"))

    report = build_report(name, cfg, headline, raw, seeds, drl_files)
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(report)

    log(f"  wrote {outdir}  ({elapsed:.0f}s)")
    return {"outdir": outdir, "summary": headline, "raw": raw, "traces": traces,
            "figures": figures, "report": report}


def _fmt(row, m, digits=3):
    return f"{row[f'{m}_mean']:.{digits}f} +/- {row[f'{m}_std']:.{digits}f}"


def build_report(name, cfg, headline, raw, seeds, drl_files) -> str:
    """Concise automated evaluation report, computed entirely from results."""
    h = headline.set_index("policy")
    order = sort_policies(list(h.index))
    L = []
    L.append(f"# Automated evaluation report - `{name}`\n")
    L.append(f"* Arrival regime: **{cfg.workload.regime}**, episode length **{cfg.episode_length}** steps, "
             f"**{cfg.facility.n_servers}** servers in **{cfg.facility.n_racks}** racks.")
    L.append(f"* Held-out evaluation seeds: {seeds} (identical workload traces for every policy).")
    L.append(f"* Reward weights: alpha={cfg.reward.alpha}, beta={cfg.reward.beta}, gamma_w={cfg.reward.gamma_w}.")
    if drl_files:
        L.append(f"* Trained DRL runs evaluated: {', '.join(drl_files)}.")
    else:
        L.append("* **No trained DRL checkpoints were found - only baseline schedulers are evaluated.**")
    L.append("")
    L.append("## Summary (mean +/- s.d. across held-out seeds)\n")
    L.append("| Policy | Hotspot freq. | Peak T (C) | Mean T (C) | IT energy (kW*step) | Cooling proxy (kW*step) | Wait (steps) | Response (steps) | Completed | Return |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in order:
        r = h.loc[p]
        L.append(f"| {p} | {_fmt(r,'hotspot_frequency')} | {_fmt(r,'peak_temperature',2)} | {_fmt(r,'mean_temperature',2)} | "
                 f"{_fmt(r,'it_energy_kwstep',0)} | {_fmt(r,'cooling_energy_proxy',0)} | {_fmt(r,'mean_wait_time',2)} | "
                 f"{_fmt(r,'mean_response_time',1)} | {_fmt(r,'tasks_completed',1)} | {_fmt(r,'episode_return',0)} |")
    L.append("")

    heur = [p for p in order if base_policy(p) in ("round-robin", "greedy-utilisation", "greedy-coolest")]
    drl = [p for p in order if base_policy(p) in ("PPO", "DQN")]
    best_heur = min(heur, key=lambda p: h.loc[p, "hotspot_frequency_mean"]) if heur else None
    L.append("## Findings\n")
    if best_heur:
        L.append(f"* Best heuristic on hotspot frequency: **{best_heur}** "
                 f"({h.loc[best_heur,'hotspot_frequency_mean']:.3f}).")
    for p in drl:
        r = h.loc[p]
        if best_heur is None:
            continue
        b = h.loc[best_heur]
        d_hot = r["hotspot_frequency_mean"] - b["hotspot_frequency_mean"]
        d_ret = r["episode_return_mean"] - b["episode_return_mean"]
        d_done = r["tasks_completed_mean"] - b["tasks_completed_mean"]
        verdict = "BETTER than" if d_hot < 0 else "WORSE than"
        L.append(f"* **{p}** hotspot frequency {r['hotspot_frequency_mean']:.3f} is {verdict} {best_heur} "
                 f"({d_hot:+.3f}); episode return {d_ret:+.0f}; tasks completed {d_done:+.1f}.")
        if "n_training_seeds" in r and not pd.isna(r.get("n_training_seeds")):
            L.append(f"  * Aggregated over {int(r['n_training_seeds'])} training seed(s); spread of hotspot "
                     f"frequency across training seeds: {r['hotspot_frequency_trainseed_std']:.3f} (s.d.).")
        if d_done < -0.02 * max(b["tasks_completed_mean"], 1):
            L.append(f"  * Caution: {p} completes fewer tasks than {best_heur}; check for a degenerate "
                     f"deferral policy before reading its thermal numbers favourably.")
    if drl and best_heur:
        any_win = any(h.loc[p, "hotspot_frequency_mean"] < h.loc[best_heur, "hotspot_frequency_mean"] for p in drl)
        if not any_win:
            L.append("* **No DRL agent beat the best heuristic on hotspot frequency in this experiment.** "
                     "Possible reasons: limited training budget; the observation omits queue length and "
                     "short-term temperature history (the formulation is only approximately Markovian); "
                     "greedy-coolest is already near-optimal when the reward is dominated by present state.")
    L.append("")
    L.append("## Notes\n")
    L.append("* Energy: IT energy = sum over steps of per-server power (idle + dynamic x utilisation). "
             "The cooling figure is a **proxy** (coefficient x hottest-server excess over ambient), not a "
             "facility cooling model; do not read it as a measured cooling saving.")
    L.append("* All figures were produced by this pipeline from `raw_results.csv`; none is hardcoded.")
    L.append("* The thermal model is a simplified lumped-parameter approximation, not CFD, and is not validated "
             "against a physical facility.")
    return "\n".join(L) + "\n"
