"""Fair-comparison guarantees and spec item 8 (metrics are calculated consistently)."""
import copy
import numpy as np
import pandas as pd
import pytest

from dcsched import (EnvConfig, DataCenterSchedulingEnv, RoundRobin, GreedyUtilisation,
                     GreedyCoolest, RandomValid)
from dcsched.evaluate import (EVAL_SEEDS, run_episode, evaluate_policy, compare_policies,
                              check_seeds_disjoint, METRIC_ORDER)


def arrival_signature(policy, seed, steps=300):
    cfg = EnvConfig(); cfg.episode_length = steps
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=seed)
    policy.reset()
    seen = []
    done = False
    # Record every task the environment ever admits, in id order.
    all_tasks = {}
    while not done:
        for t in list(env.queue):
            all_tasks[t.task_id] = (t.arrival_step, round(t.demand, 9), t.total_duration)
        for srv in env.running:
            for t in srv:
                all_tasks[t.task_id] = (t.arrival_step, round(t.demand, 9), t.total_duration)
        _, _, term, trunc, _ = env.step(policy.act(env))
        done = term or trunc
    return all_tasks


def test_every_policy_faces_the_identical_task_stream():
    sigs = [arrival_signature(p, seed=1001) for p in
            (RoundRobin(), GreedyUtilisation(), GreedyCoolest(), RandomValid(0))]
    common = set.intersection(*[set(s) for s in sigs])
    assert len(common) > 100
    for tid in common:
        assert len({s[tid] for s in sigs}) == 1, f"task {tid} differs between policies"


def test_evaluation_is_deterministic_for_deterministic_policies():
    cfg = EnvConfig(); cfg.episode_length = 300
    a = evaluate_policy(GreedyCoolest(), cfg)["per_episode"]
    b = evaluate_policy(GreedyCoolest(), cfg)["per_episode"]
    pd.testing.assert_frame_equal(a, b)


class _Poison:
    """Stands in for the workload generator; any access means a policy peeked."""
    def __getattr__(self, name):
        raise AssertionError(f"policy accessed the workload generator ({name!r}) - future leak")


def test_policies_cannot_see_the_future_arrival_stream():
    """While a policy decides, the workload generator is replaced by a booby trap.
    Baselines must still work, proving they use only the public state."""
    cfg = EnvConfig(); cfg.episode_length = 200
    for pol in (RoundRobin(), GreedyUtilisation(), GreedyCoolest(), RandomValid(0)):
        env = DataCenterSchedulingEnv(cfg)
        env.reset(seed=1001)
        pol.reset()
        real, done = env.generator, False
        while not done:
            env.generator = _Poison()
            try:
                a = pol.act(env)
            finally:
                env.generator = real
            _, _, term, trunc, _ = env.step(a)
            done = term or trunc


def test_eval_and_training_seeds_are_disjoint():
    check_seeds_disjoint([0, 1, 2])
    with pytest.raises(ValueError, match="overlap"):
        check_seeds_disjoint([0, 1003])
    assert min(EVAL_SEEDS) > 1000


def test_training_wrapper_never_draws_an_eval_seed():
    from dcsched.training import FreshTraceEnv
    env = FreshTraceEnv(DataCenterSchedulingEnv(), base_seed=0)
    drawn = {env._draw_seed() for _ in range(2000)}
    assert drawn.isdisjoint(EVAL_SEEDS)
    with pytest.raises(ValueError):
        env.reset(seed=EVAL_SEEDS[0])


# -- metric definitions (item 8) -------------------------------------------

@pytest.fixture(scope="module")
def episode():
    cfg = EnvConfig(); cfg.episode_length = 500
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=1001)
    pol = GreedyCoolest()
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(pol.act(env))
        total += r
        done = term or trunc
    return env, env.stats.summary(), total


def test_hotspot_frequency_is_steps_over_safe_divided_by_steps(episode):
    env, s, _ = episode
    safe = env.cfg.thermal.safe_temp
    manual = np.mean(np.array(env.stats.max_temp_trace) > safe)
    assert s["hotspot_frequency"] == pytest.approx(manual)
    assert s["hotspot_steps"] == int(round(manual * s["steps"]))


def test_temperature_metrics_match_recorded_traces(episode):
    env, s, _ = episode
    st = np.array(env.stats.server_temp_trace)
    assert s["peak_temperature"] == pytest.approx(st.max())
    assert s["mean_temperature"] == pytest.approx(st.mean())
    assert s["mean_max_temperature"] == pytest.approx(st.max(axis=1).mean())
    assert s["mean_spatial_variance"] == pytest.approx(st.var(axis=1).mean())


def test_throughput_and_completion_rate_definitions(episode):
    _, s, _ = episode
    assert s["throughput"] == pytest.approx(s["tasks_completed"] / s["steps"])
    assert s["completion_rate"] == pytest.approx(s["tasks_completed"] / s["tasks_arrived"])
    assert 0 <= s["completion_rate"] <= 1


def test_utilisation_metrics_match_recorded_traces(episode):
    env, s, _ = episode
    su = np.array(env.stats.server_util_trace)
    assert s["mean_utilisation"] == pytest.approx(su.mean())
    assert s["load_imbalance"] == pytest.approx(su.std(axis=1).mean())


def test_episode_return_equals_sum_of_step_rewards(episode):
    cfg = EnvConfig(); cfg.episode_length = 500
    out = run_episode(GreedyCoolest(), cfg, seed=1001)
    assert out["stats"]["episode_return"] == pytest.approx(episode[2])


def test_summary_contains_every_declared_metric():
    cfg = EnvConfig(); cfg.episode_length = 200
    s = evaluate_policy(GreedyCoolest(), cfg)["summary"]
    for m in METRIC_ORDER:
        assert f"{m}_mean" in s and f"{m}_std" in s, m


def test_aggregation_mean_and_population_std_across_seeds():
    cfg = EnvConfig(); cfg.episode_length = 200
    out = evaluate_policy(GreedyCoolest(), cfg)
    col = out["per_episode"]["hotspot_frequency"]
    assert out["summary"]["hotspot_frequency_mean"] == pytest.approx(col.mean())
    assert out["summary"]["hotspot_frequency_std"] == pytest.approx(col.std(ddof=0))
    assert len(col) == len(EVAL_SEEDS)
