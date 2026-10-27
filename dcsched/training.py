"""Resumable DRL training (WBS items 4.3, 4.4, 4.5).

Design points
-------------
* Training episodes use a *fresh workload trace each episode*, drawn from a
  training-seed stream that can never produce a held-out evaluation seed
  (``dcsched.evaluate.EVAL_SEEDS``). The agent therefore generalises over
  traces rather than memorising one, and evaluation traces are unseen.
* Checkpoints (model, learning curve, progress record) are written every
  ``checkpoint_every`` timesteps. An interrupted run resumes from the last
  checkpoint with ``resume=True``; the exploration / learning-rate schedules
  stay consistent because ``num_timesteps`` is preserved.
* The final evaluation always uses the held-out seeds.

Nothing here fabricates results: a run reports ``status`` "complete" only when
the requested number of timesteps was actually executed.
"""

import json
import os
import time
from dataclasses import asdict

import numpy as np
import gymnasium as gym

from .config import EnvConfig
from .env import DataCenterSchedulingEnv
from .evaluate import EVAL_SEEDS, evaluate_policy, check_seeds_disjoint


class FreshTraceEnv(gym.Wrapper):
    """Training wrapper: every ``reset()`` without an explicit seed draws a new
    workload seed from a private stream, never colliding with EVAL_SEEDS."""

    def __init__(self, env: DataCenterSchedulingEnv, base_seed: int):
        super().__init__(env)
        self._stream = np.random.default_rng(10_000 + int(base_seed))

    def _draw_seed(self) -> int:
        while True:
            s = int(self._stream.integers(100_000, 10_000_000))
            if s not in EVAL_SEEDS:
                return s

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self._draw_seed()
        else:
            check_seeds_disjoint([seed])
        return self.env.reset(seed=seed, options=options)


def make_training_env(cfg: EnvConfig, seed: int):
    env_cfg = EnvConfig.from_dict(cfg.to_dict())
    env_cfg.seed = seed
    return FreshTraceEnv(DataCenterSchedulingEnv(env_cfg), base_seed=seed)


def run_prefix(algo: str, seed: int) -> str:
    return f"{algo.lower()}_seed{seed}"


def _paths(outdir, algo, seed):
    p = os.path.join(outdir, run_prefix(algo, seed))
    return {
        "model": p,
        "model_zip": p + ".zip",
        "buffer": p + "_replay.pkl",
        "curve": p + "_curve.npz",
        "progress": p + "_progress.json",
        "eval": p + "_eval.json",
    }


def _read_json(path, default=None):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def train_run(algo: str, cfg: EnvConfig, seed: int, total_timesteps: int,
              outdir: str, resume: bool = True, checkpoint_every: int = 25_000,
              hyperparams: dict = None, log=print, stop_after_seconds: float = None):
    """Train one (algorithm, seed) run and evaluate it on held-out seeds.

    Returns the final progress record. If ``stop_after_seconds`` is given the
    run stops cleanly at the next checkpoint after that budget and reports
    status "partial"; call again with resume=True to continue.
    """
    from stable_baselines3.common.callbacks import BaseCallback
    from .agent import make_model, load_model, PolicyAgent

    algo = algo.upper()
    check_seeds_disjoint([seed])
    os.makedirs(outdir, exist_ok=True)
    P = _paths(outdir, algo, seed)

    env = make_training_env(cfg, seed)
    progress = _read_json(P["progress"], {}) if resume else {}
    done_steps = int(progress.get("timesteps_done", 0))
    prior_seconds = float(progress.get("train_seconds", 0.0))

    # A run whose training already finished (status "complete") but whose
    # evaluation record is missing -- e.g. the process died during evaluation --
    # must be evaluated, not retrained from scratch.
    if (resume and os.path.exists(P["model_zip"]) and done_steps >= total_timesteps
            and progress.get("status") == "complete"):
        if "eval" in progress and os.path.exists(P["eval"]):
            return progress
        model = load_model(algo, P["model"])
        agent = PolicyAgent(model, name=f"{algo}-seed{seed}")
        ev = evaluate_policy(agent, cfg, seeds=EVAL_SEEDS, name=agent.name)
        progress["eval"] = ev["summary"]
        progress["eval_seeds"] = list(EVAL_SEEDS)
        _write_json(P["eval"], progress["eval"])
        _write_json(P["progress"], progress)
        return progress

    if resume and os.path.exists(P["model_zip"]) and 0 < done_steps < total_timesteps:
        model = load_model(algo, P["model"])
        model.set_env(env)
        if algo == "DQN" and os.path.exists(P["buffer"]):
            model.load_replay_buffer(P["buffer"])
        log(f"  resumed {algo} seed {seed} from {done_steps:,} timesteps")
    else:
        done_steps, prior_seconds = 0, 0.0
        model = make_model(algo, env, seed=seed, **(hyperparams or {}))

    curve = {"returns": [], "timesteps": []}
    if resume and os.path.exists(P["curve"]) and done_steps > 0:
        old = np.load(P["curve"])
        curve = {"returns": list(old["returns"]), "timesteps": list(old["timesteps"])}

    t0 = time.time()
    state = {"stopped": False}

    def save_checkpoint(num_timesteps, status):
        model.save(P["model"])
        if algo == "DQN":
            model.save_replay_buffer(P["buffer"])
        np.savez(P["curve"], returns=np.array(curve["returns"]),
                 timesteps=np.array(curve["timesteps"]))
        _write_json(P["progress"], {
            "algo": algo, "seed": seed, "status": status,
            "timesteps_done": int(num_timesteps),
            "timesteps_target": int(total_timesteps),
            "train_seconds": prior_seconds + (time.time() - t0),
            "n_training_episodes": len(curve["returns"]),
        })

    class _CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self._current = 0.0
            self._next_ckpt = (done_steps // checkpoint_every + 1) * checkpoint_every

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            if rewards is not None:
                self._current += float(np.asarray(rewards).ravel()[0])
            if dones is not None and bool(np.asarray(dones).ravel()[0]):
                curve["returns"].append(self._current)
                curve["timesteps"].append(self.num_timesteps)
                self._current = 0.0
            if self.num_timesteps >= self._next_ckpt:
                save_checkpoint(self.num_timesteps, "running")
                self._next_ckpt += checkpoint_every
                log(f"    {algo} seed {seed}: {self.num_timesteps:,}/{total_timesteps:,} "
                    f"({time.time() - t0:.0f}s this session)")
                if stop_after_seconds and time.time() - t0 > stop_after_seconds:
                    state["stopped"] = True
                    return False
            return True

    remaining = total_timesteps - done_steps
    model.learn(total_timesteps=remaining, callback=_CB(), reset_num_timesteps=False,
                progress_bar=False)

    reached = int(model.num_timesteps)
    complete = reached >= total_timesteps and not state["stopped"]
    save_checkpoint(reached, "complete" if complete else "partial")
    final = _read_json(P["progress"])

    if complete:
        agent = PolicyAgent(model, name=f"{algo}-seed{seed}")
        ev = evaluate_policy(agent, cfg, seeds=EVAL_SEEDS, name=agent.name)
        final["eval"] = ev["summary"]
        final["eval_seeds"] = list(EVAL_SEEDS)
        _write_json(P["eval"], final["eval"])
        _write_json(P["progress"], final)
    return final


def list_runs(outdir: str):
    """Progress records for every run found under ``outdir``."""
    runs = []
    if not os.path.isdir(outdir):
        return runs
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith("_progress.json"):
            rec = _read_json(os.path.join(outdir, fn))
            if rec:
                p = os.path.join(outdir, fn.replace("_progress.json", ""))
                rec["model_path"] = p
                rec["has_checkpoint"] = os.path.exists(p + ".zip")
                ev = _read_json(p + "_eval.json")
                if ev:
                    rec["eval"] = ev
                runs.append(rec)
    return runs


def load_curve(outdir: str, algo: str, seed: int):
    f = _paths(outdir, algo, seed)["curve"]
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return {"returns": d["returns"], "timesteps": d["timesteps"]}
