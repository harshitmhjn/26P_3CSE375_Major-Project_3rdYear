"""Evaluation harness (Phase-II report, Sections 2.4, 2.5, 4.7).

All policies -- learned and baseline -- are evaluated on an identical set of
held-out workload traces generated from fixed seeds not used during training.
Each policy is run across multiple independent episodes and results are
reported as means with standard deviations, since reinforcement-learning
results are known to vary substantially across seeds.

Fairness guarantees
-------------------
* Every policy is driven through the same ``run_episode`` loop.
* The workload generator owns its RNG, so the arrival stream for a given seed
  is identical for every policy (verified in tests).
* Policies see only the public environment interface (observation / action
  mask); none sees future arrivals.
"""

import copy
import numpy as np
import pandas as pd

from .config import EnvConfig
from .env import DataCenterSchedulingEnv

# Seeds reserved for evaluation. Training seeds must lie outside this set so
# that evaluation traces are genuinely held out.
EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]

METRIC_ORDER = [
    "episode_return",
    "hotspot_frequency",
    "peak_temperature",
    "mean_temperature",
    "mean_max_temperature",
    "mean_spatial_variance",
    "total_energy",
    "it_energy_kwstep",
    "cooling_energy_proxy",
    "mean_utilisation",
    "load_imbalance",
    "mean_wait_time",
    "mean_response_time",
    "mean_queue_length",
    "throughput",
    "completion_rate",
    "tasks_arrived",
    "tasks_completed",
    "tasks_dropped",
    "defer_actions",
    "invalid_actions",
    "critical_terminations",
]


def check_seeds_disjoint(train_seeds, eval_seeds=None):
    """Raise if any training seed overlaps the held-out evaluation seeds."""
    eval_seeds = set(eval_seeds or EVAL_SEEDS)
    overlap = eval_seeds.intersection(train_seeds)
    if overlap:
        raise ValueError(f"training seeds overlap held-out evaluation seeds: {sorted(overlap)}")


def run_episode(policy, cfg: EnvConfig, seed: int, collect_traces=False):
    """Run one episode of ``policy`` against a fresh environment."""
    env = DataCenterSchedulingEnv(cfg)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    total_reward = 0.0
    done = False
    while not done:
        action = policy.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    stats = env.stats.summary()
    stats["episode_return"] = total_reward
    result = {"stats": stats}
    if collect_traces:
        result["traces"] = {
            "mean_temp": np.array(env.stats.temp_trace),
            "max_temp": np.array(env.stats.max_temp_trace),
            "utilisation": np.array(env.stats.util_trace),
            "queue": np.array(env.stats.queue_trace),
            "server_temps": np.array(env.stats.server_temp_trace),
            "server_util": np.array(env.stats.server_util_trace),
            "final_temps": env.temps.copy(),
        }
    return result


def evaluate_policy(policy, cfg: EnvConfig, seeds=None, name=None,
                    collect_traces=False):
    """Evaluate a policy across several held-out seeds."""
    seeds = seeds or EVAL_SEEDS
    name = name or getattr(policy, "name", "policy")

    per_episode = []
    traces = None
    for i, seed in enumerate(seeds):
        episode_cfg = copy.deepcopy(cfg)
        episode_cfg.seed = seed
        out = run_episode(policy, episode_cfg, seed,
                          collect_traces=collect_traces and i == 0)
        row = dict(out["stats"])
        row["seed"] = seed
        per_episode.append(row)
        if "traces" in out:
            traces = out["traces"]

    df = pd.DataFrame(per_episode)
    summary = {"policy": name}
    for metric in METRIC_ORDER:
        if metric in df.columns:
            summary[f"{metric}_mean"] = float(df[metric].mean())
            summary[f"{metric}_std"] = float(df[metric].std(ddof=0))
    return {"summary": summary, "per_episode": df, "traces": traces}


def compare_policies(policies, cfg: EnvConfig, seeds=None, collect_traces=False,
                     return_raw=False):
    """Evaluate several policies under identical conditions.

    Returns (summary_df, traces) or, with ``return_raw``, (summary_df, traces,
    raw_df) where raw_df holds one row per (policy, seed).
    """
    rows, all_traces, raws = [], {}, []
    for policy in policies:
        name = getattr(policy, "name", str(policy))
        out = evaluate_policy(policy, cfg, seeds=seeds, name=name,
                              collect_traces=collect_traces)
        rows.append(out["summary"])
        raw = out["per_episode"].copy()
        raw.insert(0, "policy", name)
        raws.append(raw)
        if out["traces"] is not None:
            all_traces[name] = out["traces"]
    summary = pd.DataFrame(rows)
    if return_raw:
        return summary, all_traces, pd.concat(raws, ignore_index=True)
    return summary, all_traces


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Condense the full results frame to the metrics worth showing in a viva."""
    cols = {
        "policy": "Policy",
        "hotspot_frequency_mean": "Hotspot freq.",
        "peak_temperature_mean": "Peak temp (C)",
        "mean_temperature_mean": "Mean temp (C)",
        "total_energy_mean": "Energy",
        "mean_wait_time_mean": "Wait (steps)",
        "tasks_completed_mean": "Completed",
        "episode_return_mean": "Return",
    }
    present = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(present)].rename(columns=present)
    return out.round(3)
