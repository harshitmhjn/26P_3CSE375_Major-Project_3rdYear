"""Spec item 5 - baseline schedulers produce valid decisions through a common interface."""
import numpy as np
import pytest

from dcsched import (DataCenterSchedulingEnv, EnvConfig, RoundRobin, GreedyUtilisation,
                     GreedyCoolest, RandomValid, BASELINES, make_baseline)

ALL = [RoundRobin, GreedyUtilisation, GreedyCoolest, RandomValid]


@pytest.mark.parametrize("cls", ALL)
def test_baselines_only_emit_valid_actions(cls):
    env = DataCenterSchedulingEnv()
    env.reset(seed=13)
    pol = cls()
    pol.reset()
    done = False
    while not done:
        a = pol.act(env)
        assert env.action_mask()[a], f"{cls.__name__} chose infeasible action {a}"
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
    assert env.stats.invalid_actions == 0


@pytest.mark.parametrize("cls", ALL)
def test_baselines_defer_when_nothing_is_pending(cls):
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    env.queue.clear()
    assert cls().act(env) == env.DEFER


@pytest.mark.parametrize("cls", ALL)
def test_baselines_defer_when_no_server_can_accept(cls):
    env = DataCenterSchedulingEnv()
    env.reset(seed=1)
    while env._pending() is None:
        env.step(env.DEFER)
    env.alloc[:] = env.cfg.facility.server_capacity     # every server full
    assert not env.action_mask()[: env.n].any()
    assert cls().act(env) == env.DEFER


def test_round_robin_cycles_through_servers():
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    rr = RoundRobin()
    rr.reset()
    picks = []
    for _ in range(400):
        a = rr.act(env)
        if a != env.DEFER:
            picks.append(a)
        env.step(a)
        if len(picks) >= 8:
            break
    # consecutive placements advance around the ring (mod n) while capacity is free
    assert picks[:6] == [(picks[0] + i) % env.n for i in range(6)]


def test_greedy_utilisation_picks_the_least_loaded_feasible_server():
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    while env._pending() is None:
        env.step(env.DEFER)
    env.alloc[:] = np.linspace(0.1, 0.55, env.n)[::-1]   # server n-1 is least loaded
    mask = env.action_mask()
    expect = int(np.argmin(np.where(mask[: env.n], env.alloc, np.inf)))
    assert GreedyUtilisation().act(env) == expect


def test_greedy_coolest_picks_the_coolest_feasible_server():
    env = DataCenterSchedulingEnv()
    env.reset(seed=0)
    while env._pending() is None:
        env.step(env.DEFER)
    env.temps[:] = np.linspace(60, 35, env.n)            # coolest is the last server
    mask = env.action_mask()
    expect = int(np.argmin(np.where(mask[: env.n], env.temps, np.inf)))
    assert GreedyCoolest().act(env) == expect


def test_thermal_ordering_of_baselines_on_held_out_seeds():
    """Sanity: informed policies beat naive ones on hotspot frequency."""
    from dcsched.evaluate import compare_policies
    cfg = EnvConfig(); cfg.episode_length = 600
    df, _ = compare_policies([RandomValid(0), RoundRobin(), GreedyUtilisation(), GreedyCoolest()], cfg)
    h = df.set_index("policy")["hotspot_frequency_mean"]
    assert h["greedy-coolest"] < h["round-robin"]
    assert h["greedy-utilisation"] < h["round-robin"]


def test_make_baseline_registry_and_unknown_name():
    for name in BASELINES:
        assert make_baseline(name).name == name
    with pytest.raises(ValueError, match="unknown baseline"):
        make_baseline("does-not-exist")
