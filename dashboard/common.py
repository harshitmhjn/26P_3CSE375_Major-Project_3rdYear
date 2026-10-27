"""Shared state, configuration and helpers for the dashboard pages."""

import os
import subprocess
import sys
import time

import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dcsched import EnvConfig                                    # noqa: E402
from dcsched.config import FacilityConfig, ThermalConfig, WorkloadConfig, RewardConfig  # noqa: E402
from dcsched.live import LiveSimulation                          # noqa: E402
from dcsched.policies import drl_options, build_policy, heuristic_names  # noqa: E402
from dcsched.evaluate import EVAL_SEEDS                          # noqa: E402

RESULTS = os.path.join(ROOT, "results")
EXPERIMENTS_DIR = os.path.join(RESULTS, "experiments")
MODELS_DIR = os.path.join(RESULTS, "models")
UI_MODELS_DIR = os.path.join(RESULTS, "models_ui")

DEFAULT_SIM = dict(
    n_servers=16, n_racks=4, regime="steady", arrival_rate=0.9, episode_length=1000,
    seed=1001, queue_capacity=12, alpha=1.0, beta=0.5, gamma_w=0.3,
    heat_per_util=4.0, cooling_rate=0.22, safe_temp=58.0, policy="greedy-coolest",
)


def make_cfg(p: dict) -> EnvConfig:
    """Build an EnvConfig from UI parameters. Raises ValueError on invalid input
    (the backend's own validation is the single source of truth)."""
    thermal = ThermalConfig(heat_per_util=p["heat_per_util"], cooling_rate=p["cooling_rate"],
                            safe_temp=p["safe_temp"])
    return EnvConfig(
        facility=FacilityConfig(n_servers=p["n_servers"], n_racks=p["n_racks"]),
        thermal=thermal,
        workload=WorkloadConfig(regime=p["regime"], arrival_rate=p["arrival_rate"],
                                queue_capacity=p["queue_capacity"]),
        reward=RewardConfig(alpha=p["alpha"], beta=p["beta"], gamma_w=p["gamma_w"]),
        episode_length=p["episode_length"], seed=p["seed"],
    )


def drl_runs():
    """Completed DRL checkpoints available on disk (re-read each call so newly
    finished training runs appear without restarting the app)."""
    return drl_options((MODELS_DIR, UI_MODELS_DIR))


def get_policy(name: str, cfg: EnvConfig):
    """Heuristics are built fresh (round-robin is stateful); DRL models are
    cached per label because loading them is slow."""
    runs = drl_runs()
    if name in runs:
        cache = st.session_state.setdefault("drl_cache", {})
        key = (name, runs[name]["model_path"], cfg.facility.n_servers)
        if key not in cache:
            cache[key] = build_policy(name, runs, cfg=cfg)
        return cache[key]
    return build_policy(name, runs, cfg=cfg)


def init_state():
    ss = st.session_state
    if "sim_params" not in ss:
        ss.sim_params = dict(DEFAULT_SIM)
    if "sim" not in ss:
        cfg = make_cfg(ss.sim_params)
        ss.sim = LiveSimulation(cfg, get_policy(ss.sim_params["policy"], cfg), seed=cfg.seed)
    ss.setdefault("auto_run", False)
    ss.setdefault("tick_steps", 5)
    ss.setdefault("policy_error", None)
    ss.setdefault("last_comparison", None)
    ss.setdefault("train_proc", None)


def rebuild_simulation(params: dict):
    """Apply new parameters and reset. Returns an error string or None."""
    ss = st.session_state
    try:
        cfg = make_cfg(params)
        policy = get_policy(params["policy"], cfg)
    except (ValueError, FileNotFoundError) as exc:
        return str(exc)
    ss.sim_params = dict(params)
    ss.sim = LiveSimulation(cfg, policy, seed=cfg.seed)
    ss.auto_run = False
    return None


# -- background training launched from the dashboard -------------------------

def launch_training(algo, timesteps, seed, regime, alpha, beta, gamma_w):
    """Start scripts/train_agent.py as a separate process (works on Windows too)
    writing into its own folder under results/models_ui/."""
    if seed in EVAL_SEEDS:
        raise ValueError(f"training seed {seed} is reserved for held-out evaluation {EVAL_SEEDS}")
    if timesteps < 2000:
        raise ValueError("use at least 2,000 timesteps")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{algo.lower()}_s{seed}"
    outdir = os.path.join(UI_MODELS_DIR, run_id)
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "train.log"), "w")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_agent.py"),
           "--algo", algo, "--timesteps", str(int(timesteps)), "--seeds", str(int(seed)),
           "--regime", regime, "--outdir", outdir,
           "--checkpoint-every", str(max(500, int(timesteps) // 10)),
           "--alpha", str(alpha), "--beta", str(beta), "--gamma-w", str(gamma_w)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return {"proc": proc, "outdir": outdir, "run_id": run_id, "algo": algo,
            "seed": int(seed), "timesteps": int(timesteps)}


def tail(path, n=12):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as fh:
        return "".join(fh.readlines()[-n:])


STATUS_ICON = {"normal": "🟢", "warm": "🟡", "hotspot": "🔴", "critical": "⛔"}
