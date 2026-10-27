"""Spec items 7, 9, 11 - checkpoints, result export, and graceful handling of bad input."""
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from dcsched import EnvConfig, DataCenterSchedulingEnv
from dcsched.agent import make_model, load_model, PolicyAgent, default_hyperparameters
from dcsched.training import train_run, list_runs, load_curve, make_training_env
from dcsched.experiments import run_experiment, discover_agents
from dcsched.evaluate import EVAL_SEEDS, evaluate_policy


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = EnvConfig(); cfg.episode_length = 150
    return cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_cfg):
    out = str(tmp_path_factory.mktemp("models"))
    train_run("PPO", tiny_cfg, 0, 1024, out, checkpoint_every=512, log=lambda *a: None)
    train_run("DQN", tiny_cfg, 0, 1500, out, checkpoint_every=500,
              hyperparams={"learning_starts": 300}, log=lambda *a: None)
    return out


# -- checkpoints (item 7) --------------------------------------------------

@pytest.mark.parametrize("algo", ["PPO", "DQN"])
def test_checkpoint_roundtrip_gives_identical_predictions(trained, tiny_cfg, algo):
    rec = [r for r in list_runs(trained) if r["algo"] == algo][0]
    assert rec["status"] == "complete" and rec["has_checkpoint"]
    m1 = load_model(algo, rec["model_path"])
    m2 = load_model(algo, rec["model_path"])
    env = DataCenterSchedulingEnv(tiny_cfg)
    obs, _ = env.reset(seed=1001)
    for _ in range(20):
        a1, _ = m1.predict(obs, deterministic=True)
        a2, _ = m2.predict(obs, deterministic=True)
        assert int(a1) == int(a2)
        obs, *_ = env.step(int(a1) if env.action_mask()[int(a1)] else env.DEFER)


def test_training_actually_records_progress_and_curve(trained):
    rec = list_runs(trained)[0]
    assert rec["timesteps_done"] >= rec["timesteps_target"] > 0
    curve = load_curve(trained, rec["algo"], rec["seed"])
    assert curve is not None and len(curve["returns"]) == len(curve["timesteps"]) > 0


def test_training_run_is_evaluated_on_heldout_seeds(trained):
    for rec in list_runs(trained):
        assert rec["eval_seeds"] == EVAL_SEEDS
        assert "hotspot_frequency_mean" in rec["eval"]


def test_interrupted_training_resumes_from_checkpoint(tmp_path, tiny_cfg):
    out = str(tmp_path)
    r1 = train_run("PPO", tiny_cfg, 1, 2048, out, checkpoint_every=1024,
                   stop_after_seconds=1e-6, log=lambda *a: None)
    assert r1["status"] == "partial" and r1["timesteps_done"] < 2048
    assert "eval" not in r1                       # a partial run is never presented as trained
    r2 = train_run("PPO", tiny_cfg, 1, 2048, out, checkpoint_every=1024, log=lambda *a: None)
    assert r2["status"] == "complete" and r2["timesteps_done"] >= 2048 and "eval" in r2


def test_completed_run_without_eval_record_is_evaluated_not_retrained(tmp_path, tiny_cfg):
    out = str(tmp_path)
    r = train_run("PPO", tiny_cfg, 2, 1024, out, checkpoint_every=512, log=lambda *a: None)
    mtime = os.path.getmtime(os.path.join(out, "ppo_seed2.zip"))
    os.remove(os.path.join(out, "ppo_seed2_eval.json"))
    p = os.path.join(out, "ppo_seed2_progress.json")
    rec = json.load(open(p)); rec.pop("eval", None); json.dump(rec, open(p, "w"))
    r2 = train_run("PPO", tiny_cfg, 2, 1024, out, checkpoint_every=512, log=lambda *a: None)
    assert "eval" in r2
    assert os.path.getmtime(os.path.join(out, "ppo_seed2.zip")) == mtime   # model untouched


def test_policy_agent_never_returns_infeasible_action(trained, tiny_cfg):
    for algo in ("PPO", "DQN"):
        rec = [r for r in list_runs(trained) if r["algo"] == algo][0]
        agent = PolicyAgent(load_model(algo, rec["model_path"]), name=algo)
        env = DataCenterSchedulingEnv(tiny_cfg)
        env.reset(seed=1002)
        done = False
        while not done:
            a = agent.act(env)
            assert env.action_mask()[a]
            _, _, term, trunc, _ = env.step(a)
            done = term or trunc
        assert env.stats.invalid_actions == 0


def test_default_hyperparameters_exposed():
    for algo in ("PPO", "DQN"):
        hp = default_hyperparameters(algo)
        assert hp["net_arch"] == [128, 128] and 0 < hp["gamma"] <= 1


# -- export (item 9) -------------------------------------------------------

def test_experiment_exports_csv_config_report_and_figures(tmp_path, trained, tiny_cfg):
    res = run_experiment("e2e", tiny_cfg, outdir_root=str(tmp_path), models_dir=trained,
                         log=lambda *a: None)
    d = res["outdir"]
    for f in ("raw_results.csv", "summary.csv", "summary_per_run.csv", "config.json", "report.md"):
        assert os.path.getsize(os.path.join(d, f)) > 0, f
    for png in ("thermal_summary", "energy_work", "performance", "temperature_trace", "server_heatmaps"):
        assert os.path.getsize(os.path.join(d, "figures", png + ".png")) > 1000, png
    raw = pd.read_csv(os.path.join(d, "raw_results.csv"))
    pols = set(raw["policy"])
    assert {"random", "round-robin", "greedy-utilisation", "greedy-coolest"} <= pols
    assert any(p.startswith("PPO-seed") for p in pols) and any(p.startswith("DQN-seed") for p in pols)
    assert len(raw) == len(pols) * len(EVAL_SEEDS)
    cfgj = json.load(open(os.path.join(d, "config.json")))
    assert cfgj["evaluation_seeds"] == EVAL_SEEDS
    assert cfgj["environment_config"]["episode_length"] == tiny_cfg.episode_length


def test_summary_csv_matches_raw_rows(tmp_path, tiny_cfg):
    res = run_experiment("nodrl", tiny_cfg, outdir_root=str(tmp_path), include_drl=False,
                         make_figures=False, log=lambda *a: None)
    raw = pd.read_csv(os.path.join(res["outdir"], "raw_results.csv"))
    summ = pd.read_csv(os.path.join(res["outdir"], "summary.csv")).set_index("policy")
    for pol, g in raw.groupby("policy"):
        assert summ.loc[pol, "hotspot_frequency_mean"] == pytest.approx(g["hotspot_frequency"].mean())
        assert summ.loc[pol, "episode_return_mean"] == pytest.approx(g["episode_return"].mean())


def test_report_is_honest_when_no_drl_present(tmp_path, tiny_cfg):
    res = run_experiment("nodrl2", tiny_cfg, outdir_root=str(tmp_path), include_drl=False,
                         make_figures=False, log=lambda *a: None)
    assert "No trained DRL checkpoints" in res["report"]


# -- invalid input / unavailable checkpoints (item 11) ---------------------

def test_loading_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_agent"):
        load_model("PPO", str(tmp_path / "nope"))


def test_unknown_algorithm_is_rejected(tmp_path, tiny_cfg):
    with pytest.raises(ValueError):
        make_model("A2C", make_training_env(tiny_cfg, 0))
    with pytest.raises(ValueError):
        load_model("SAC", str(tmp_path / "x"))
    with pytest.raises(ValueError):
        default_hyperparameters("nope")


def test_training_with_eval_seed_is_refused(tmp_path, tiny_cfg):
    with pytest.raises(ValueError, match="overlap"):
        train_run("PPO", tiny_cfg, 1001, 512, str(tmp_path), log=lambda *a: None)


def test_discover_agents_skips_unfinished_runs(tmp_path, tiny_cfg):
    out = str(tmp_path)
    train_run("PPO", tiny_cfg, 0, 2048, out, checkpoint_every=1024,
              stop_after_seconds=1e-6, log=lambda *a: None)
    assert discover_agents(out) == []             # partial run is not presented as a trained agent


def test_empty_or_missing_models_dir_is_handled(tmp_path):
    assert list_runs(str(tmp_path / "missing")) == []
    assert discover_agents(str(tmp_path)) == []
    assert load_curve(str(tmp_path), "PPO", 0) is None
