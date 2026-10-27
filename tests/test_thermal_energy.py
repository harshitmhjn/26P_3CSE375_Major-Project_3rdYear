"""Spec item 4 - temperatures and energy behave consistently with the documented models."""
import numpy as np
import pytest

from dcsched import EnvConfig, ThermalModel, DataCenterSchedulingEnv, GreedyCoolest
from dcsched.config import FacilityConfig, ThermalConfig


def _model(**kw):
    cfg = EnvConfig()
    for k, v in kw.items():
        setattr(cfg.thermal, k, v)
    return cfg, ThermalModel(cfg.facility, cfg.thermal)


def test_idle_facility_is_an_exact_fixed_point():
    cfg, m = _model()
    t = m.initial_temperatures()
    n = cfg.facility.n_servers
    for _ in range(200):
        t = m.step(t, np.zeros(n), np.ones(n))
    assert np.allclose(t, cfg.thermal.idle_temp)


def test_update_rule_matches_documented_equation_for_one_server():
    """Single hot server: check T(t+1) against the equation in thermal.py."""
    cfg, m = _model()
    n = cfg.facility.n_servers
    temps = np.full(n, cfg.thermal.idle_temp)
    temps[0] = 50.0
    util = np.zeros(n)
    util[0] = 0.5
    k = np.ones(n)
    out = m.step(temps, util, k)
    c = cfg.thermal
    excess = np.maximum(temps - c.idle_temp, 0)
    expected = temps + c.heat_per_util * util * k + m._coupling @ excess \
        - c.cooling_rate * (temps - c.idle_temp)
    expected = np.clip(expected, c.ambient_temp, None)
    assert np.allclose(out, expected)


def test_load_raises_temperature_monotonically_toward_equilibrium():
    cfg, m = _model()
    n = cfg.facility.n_servers
    t = m.initial_temperatures()
    last = t.mean()
    for _ in range(40):
        t = m.step(t, np.full(n, 0.6), np.ones(n))
        assert t.mean() >= last - 1e-9
        last = t.mean()
    assert last > cfg.thermal.idle_temp + 5


def test_temperature_never_below_ambient():
    cfg, m = _model()
    n = cfg.facility.n_servers
    t = np.full(n, cfg.thermal.ambient_temp)
    for _ in range(50):
        t = m.step(t, np.zeros(n), np.ones(n))
        assert np.all(t >= cfg.thermal.ambient_temp - 1e-9)


def test_higher_thermal_intensity_gives_hotter_server():
    cfg, m = _model()
    n = cfg.facility.n_servers
    cool, hot = m.initial_temperatures(), m.initial_temperatures()
    util = np.full(n, 0.5)
    for _ in range(60):
        cool = m.step(cool, util, np.full(n, 0.6))
        hot = m.step(hot, util, np.full(n, 1.4))
    assert hot.mean() > cool.mean() + 2


def test_placement_matters_concentrated_load_is_hotter_than_spread():
    """The property the whole project rests on."""
    cfg, m = _model()
    n, spr = cfg.facility.n_servers, cfg.facility.servers_per_rack
    conc = np.zeros(n); conc[:spr] = 1.0
    spread = np.full(n, spr / n)
    assert conc.sum() == pytest.approx(spread.sum())
    tc, ts = m.initial_temperatures(), m.initial_temperatures()
    ones = np.ones(n)
    for _ in range(150):
        tc = m.step(tc, conc, ones)
        ts = m.step(ts, spread, ones)
    assert tc.max() > ts.max() + 3.0


def test_same_rack_coupling_exceeds_cross_rack_and_diagonal_is_zero():
    cfg, m = _model()
    spr = cfg.facility.servers_per_rack
    assert np.allclose(np.diag(m._coupling), 0)
    assert m._coupling[0, 1] > m._coupling[0, spr]


def test_unstable_thermal_configuration_is_rejected():
    cfg = EnvConfig()
    cfg.thermal.recirc_same_rack = 0.2   # gain far above the cooling rate
    with pytest.raises(ValueError, match="unstable"):
        ThermalModel(cfg.facility, cfg.thermal)


def test_facility_config_validation():
    with pytest.raises(ValueError):
        FacilityConfig(n_servers=10, n_racks=4)
    with pytest.raises(ValueError):
        FacilityConfig(n_servers=0)
    with pytest.raises(ValueError):
        FacilityConfig(server_capacity=0)


# -- energy model -----------------------------------------------------------

def test_server_power_follows_idle_plus_dynamic_model():
    env = DataCenterSchedulingEnv()
    env.reset(seed=1)
    r = env.cfg.reward
    # empty facility draws no IT power (servers with no task are 'off/idle-free')
    assert np.allclose(env.server_power(), 0.0)
    # place a task on server 3 manually via the policy interface
    while env._pending() is None:
        env.step(env.DEFER)
    d = env._pending().demand
    env.step(3)
    p = env.server_power()
    # the task may or may not have finished within the step; it has duration >= 12
    assert p[3] == pytest.approx(r.idle_power + r.dynamic_power * d)
    assert np.allclose(np.delete(p, 3), 0.0)


def test_energy_accumulates_consistently_over_time():
    env = DataCenterSchedulingEnv()
    env.reset(seed=5)
    pol = GreedyCoolest()
    it_sum, cool_sum, done = 0.0, 0.0, False
    for _ in range(200):
        _, _, term, trunc, info = env.step(pol.act(env))
        it_sum += info["it_power_kw"]
        cool_sum += info["cooling_power_kw"]
        if term or trunc:
            break
    s = env.stats.summary()
    assert s["it_energy_kwstep"] == pytest.approx(it_sum)
    assert s["cooling_energy_proxy"] == pytest.approx(cool_sum)
    r = env.cfg.reward
    assert s["total_energy"] == pytest.approx((it_sum + cool_sum) / r.energy_scale)


def test_cooling_proxy_includes_idle_offset_above_ambient():
    env = DataCenterSchedulingEnv()
    env.reset(seed=1)
    it, cool = env._energy_terms()
    # idle temp (30) exceeds ambient (22): proxy is coeff * 8, NOT zero. Document it.
    assert it == 0.0
    assert cool == pytest.approx(env.cfg.reward.cooling_coeff * (30.0 - 22.0))
