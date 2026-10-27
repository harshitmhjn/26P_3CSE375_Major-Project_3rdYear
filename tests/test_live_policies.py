"""Backend support for the dashboard: live stepping, pausing, reset, policy registry."""
import pytest

from dcsched import EnvConfig, GreedyCoolest, RoundRobin
from dcsched.live import LiveSimulation
from dcsched.policies import build_policy, drl_options, heuristic_names


def cfg(n=120):
    c = EnvConfig(); c.episode_length = n
    return c


def test_stepping_in_chunks_equals_one_long_run():
    a = LiveSimulation(cfg(), GreedyCoolest(), seed=1001)
    b = LiveSimulation(cfg(), GreedyCoolest(), seed=1001)
    a.step(120)
    for _ in range(12):
        b.step(10)
    assert a.summary() == b.summary()
    assert a.cumulative_reward == pytest.approx(b.cumulative_reward)


def test_pause_between_steps_does_not_change_results():
    """Pausing = simply not calling step(); resuming must be seamless."""
    a = LiveSimulation(cfg(), GreedyCoolest(), seed=3)
    a.step(40); snap = a.summary(); a.step(0); assert a.summary() == snap
    a.step(80)
    b = LiveSimulation(cfg(), GreedyCoolest(), seed=3); b.step(120)
    assert a.summary() == b.summary()


def test_stops_at_episode_end_and_records_history():
    s = LiveSimulation(cfg(60), RoundRobin(), seed=2)
    assert s.step(500) == 60 and s.done
    assert s.step(10) == 0
    assert len(s.history) == 60 and s.history[-1]["step"] == 60
    assert {"mean_temp", "max_temp", "queue", "completed", "hotspot"} <= set(s.history[0])


def test_reset_restores_initial_state_and_new_seed_changes_trace():
    s = LiveSimulation(cfg(), GreedyCoolest(), seed=5)
    s.step(50); first = s.summary()
    s.reset()
    assert s.step_count == 0 and s.history == [] and s.cumulative_reward == 0
    s.step(50)
    assert s.summary() == first
    s.reset(seed=6); s.step(50)
    assert s.summary() != first


def test_policy_can_be_swapped_mid_run():
    s = LiveSimulation(cfg(), RoundRobin(), seed=1)
    s.step(30); s.set_policy(GreedyCoolest()); s.step(30)
    assert s.step_count == 60


def test_snapshot_and_queue_views_have_expected_fields():
    s = LiveSimulation(cfg(), GreedyCoolest(), seed=1)
    s.step(80)
    snap = s.snapshot()
    assert len(snap) == s.cfg.facility.n_servers
    assert {"server", "rack", "cpu_util", "temperature_c", "power_kw", "status", "task_ids"} <= set(snap[0])
    assert all(r["status"] in ("normal", "warm", "hotspot", "critical") for r in snap)
    for q in s.queue_view():
        assert q["waiting_steps"] >= 0


def test_policy_registry():
    for n in heuristic_names():
        assert build_policy(n).name == n
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy("PPO-seed0", drl={})
    assert drl_options(("this/does/not/exist",)) == {}
