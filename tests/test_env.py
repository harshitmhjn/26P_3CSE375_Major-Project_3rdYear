"""Spec items 2, 3, 6 - capacity constraints, task lifecycle, Gymnasium interface."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from dcsched import EnvConfig, DataCenterSchedulingEnv, GreedyUtilisation, GreedyCoolest


def run(env, policy, steps=None):
    done = False
    n = 0
    while not done and (steps is None or n < steps):
        _, _, term, trunc, _ = env.step(policy.act(env))
        done = term or trunc
        n += 1


# -- Gymnasium interface (item 6) ------------------------------------------

def test_observation_and_action_spaces_match_the_design():
    env = DataCenterSchedulingEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (2 * env.n + 5,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.n == env.n + 1 and env.DEFER == env.n
    assert info == {}


def test_step_returns_the_five_gymnasium_values():
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    out = env.step(env.DEFER)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert {"action_mask", "invalid_action", "thermal_risk",
            "it_power_kw", "cooling_power_kw"} <= set(info)


def test_gymnasium_env_checker_passes():
    cfg = EnvConfig(); cfg.episode_length = 50
    check_env(DataCenterSchedulingEnv(cfg), skip_render_check=True)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        DataCenterSchedulingEnv().step(0)


@pytest.mark.parametrize("bad", [-1, 17, 100])
def test_out_of_range_action_raises(bad):
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(bad)


def test_episode_truncates_at_configured_length_and_reports_stats():
    cfg = EnvConfig(); cfg.episode_length = 120
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=0)
    n = 0
    while True:
        _, _, term, trunc, info = env.step(env.DEFER)
        n += 1
        if term or trunc:
            break
    assert n == 120 and trunc and not term
    assert info["episode_stats"]["steps"] == 120


def test_critical_overheat_terminates_with_penalty():
    cfg = EnvConfig(); cfg.episode_length = 600
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=0)
    env.temps[:] = cfg.thermal.critical_temp + 5     # force an overheat
    _, reward, term, _, _ = env.step(env.DEFER)
    assert term
    assert reward <= -cfg.reward.critical_penalty
    assert env.stats.critical_terminations == 1


def test_reset_is_clean_and_reproducible():
    env = DataCenterSchedulingEnv()
    o1, _ = env.reset(seed=9)
    run(env, GreedyCoolest(), steps=100)
    o2, _ = env.reset(seed=9)
    assert np.array_equal(o1, o2)
    assert env.step_count == 0 and env.stats.completed == 0
    assert env.alloc.sum() == 0


# -- capacity constraints (item 2) -----------------------------------------

def test_capacity_never_exceeded_under_random_valid_actions():
    env = DataCenterSchedulingEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    cap = env.cfg.facility.server_capacity
    for _ in range(700):
        valid = np.flatnonzero(env.action_mask())
        env.step(int(rng.choice(valid)))
        assert np.all(env.alloc <= cap + 1e-6)


def test_action_mask_is_exactly_the_feasible_set():
    env = DataCenterSchedulingEnv()
    env.reset(seed=5)
    cap = env.cfg.facility.server_capacity
    pol = GreedyUtilisation()
    for _ in range(300):
        task = env._pending()
        mask = env.action_mask()
        assert mask[env.DEFER]
        if task is None:
            assert mask.sum() == 1
        else:
            for i in range(env.n):
                assert mask[i] == (env.alloc[i] + task.demand <= cap + 1e-9)
        env.step(pol.act(env))


def test_invalid_action_is_executed_as_defer_and_penalised():
    cfg = EnvConfig()
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=1)
    # fill server 0 completely, then try to place on it again
    while env._pending() is None:
        env.step(env.DEFER)
    env.alloc[0] = cfg.facility.server_capacity
    assert not env.action_mask()[0]
    q_before = len(env.queue)
    _, reward, _, _, info = env.step(0)
    assert info["invalid_action"] is True
    assert env.stats.invalid_actions == 1
    assert len(env.queue) >= q_before          # nothing was placed
    assert reward < 0


# -- task lifecycle (item 3) ------------------------------------------------

def test_task_lifecycle_arrived_queued_running_completed():
    cfg = EnvConfig(); cfg.episode_length = 400
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=2)
    while env._pending() is None:
        env.step(env.DEFER)
    task = env._pending()
    assert task.server == -1 and task.start_step == -1
    t_place = env.step_count
    env.step(4)
    assert task.server == 4 and task.start_step == t_place
    assert task in env.running[4] and task not in env.queue
    # run until it finishes
    for _ in range(task.total_duration + 2):
        env.step(env.DEFER)
        if task.finish_step != -1:
            break
    assert task.finish_step != -1
    assert task.finish_step - task.start_step == task.total_duration
    assert task not in env.running[4]
    assert env.stats.completed >= 1


def test_wait_and_response_times_are_consistent():
    """response = wait + duration, per task and therefore in aggregate."""
    cfg = EnvConfig(); cfg.episode_length = 500
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=6)
    finished = []
    pol = GreedyCoolest()
    done = False
    while not done:
        before = [t for srv in env.running for t in srv]
        _, _, term, trunc, _ = env.step(pol.act(env))
        done = term or trunc
        finished += [t for t in before if t.finish_step != -1]
    assert len(finished) > 50
    for t in finished:
        wait = t.start_step - t.arrival_step
        assert wait >= 0
        assert t.finish_step - t.arrival_step == wait + t.total_duration
    s = env.stats.summary()
    assert s["mean_response_time"] >= s["mean_wait_time"]


def test_task_conservation_arrived_equals_dropped_plus_placed_plus_queued():
    cfg = EnvConfig(); cfg.episode_length = 600
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=8)
    run(env, GreedyUtilisation())
    s = env.stats
    assert s.arrived == s.dropped + s.placed + len(env.queue)
    running = sum(len(r) for r in env.running)
    assert s.placed == s.completed + running


def test_queue_overflow_drops_tasks_and_bills_them():
    cfg = EnvConfig(); cfg.episode_length = 200
    env = DataCenterSchedulingEnv(cfg)
    env.reset(seed=4)
    for _ in range(200):
        env.step(env.DEFER)
    assert env.stats.dropped > 0
    assert env.stats.completed == 0


def test_always_defer_is_a_losing_policy_not_an_exploit():
    """Regression test for the reward-hacking failure found during development."""
    cfg = EnvConfig(); cfg.episode_length = 400
    returns = {}
    for name, pol in (("defer", None), ("work", GreedyCoolest())):
        env = DataCenterSchedulingEnv(cfg)
        env.reset(seed=4)
        total, done = 0.0, False
        while not done:
            a = env.DEFER if pol is None else pol.act(env)
            _, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns[name] = total
    assert returns["defer"] < returns["work"]
