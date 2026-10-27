"""Spec item 1 - workloads are generated with valid parameters (and reproducibly)."""
import numpy as np
import pytest

from dcsched import EnvConfig, WorkloadGenerator, WorkloadConfig


def _collect(cfg, seed, steps=400):
    gen = WorkloadGenerator(cfg, seed=seed)
    return [t for s in range(steps) for t in gen.arrivals(s)]


def test_task_parameters_are_within_configured_ranges():
    cfg = WorkloadConfig()
    tasks = _collect(cfg, seed=3, steps=600)
    assert len(tasks) > 200
    for t in tasks:
        assert cfg.demand_min <= t.demand <= cfg.demand_max
        assert cfg.duration_min <= t.duration <= cfg.duration_max
        assert cfg.thermal_min <= t.thermal_intensity <= cfg.thermal_max
        assert t.category in ("compute-bound", "io-bound")
        assert t.total_duration == t.duration
        assert t.start_step == -1 and t.finish_step == -1 and t.server == -1


def test_task_ids_are_unique_and_sequential():
    tasks = _collect(WorkloadConfig(), seed=1)
    ids = [t.task_id for t in tasks]
    assert ids == list(range(len(ids)))


def test_arrival_steps_are_non_decreasing():
    tasks = _collect(WorkloadConfig(), seed=2)
    steps = [t.arrival_step for t in tasks]
    assert steps == sorted(steps)


def test_same_seed_same_trace_and_different_seed_differs():
    cfg = WorkloadConfig()
    a = [t.as_features() for t in _collect(cfg, 42)]
    b = [t.as_features() for t in _collect(cfg, 42)]
    c = [t.as_features() for t in _collect(cfg, 43)]
    assert a == b
    assert a != c


@pytest.mark.parametrize("regime", ["steady", "diurnal", "bursty"])
def test_every_regime_produces_work(regime):
    cfg = WorkloadConfig(regime=regime)
    assert len(_collect(cfg, seed=7, steps=800)) > 100


def test_mean_arrival_rate_matches_configuration():
    cfg = WorkloadConfig(arrival_rate=0.9)
    n = len(_collect(cfg, seed=11, steps=5000))
    assert n / 5000 == pytest.approx(0.9, rel=0.08)


def test_bursty_regime_is_burstier_than_steady():
    def per_step_counts(regime):
        gen = WorkloadGenerator(WorkloadConfig(regime=regime), seed=5)
        return np.array([len(gen.arrivals(s)) for s in range(3000)])
    assert per_step_counts("bursty").var() > per_step_counts("steady").var()


def test_diurnal_rate_oscillates():
    cfg = WorkloadConfig(regime="diurnal")
    gen = WorkloadGenerator(cfg, seed=0)
    rates = [gen._rate_at(s) for s in range(cfg.diurnal_period)]
    assert max(rates) > 1.5 * min(rates)


@pytest.mark.parametrize("bad", [
    dict(regime="nonsense"),
    dict(demand_min=0.0),
    dict(demand_min=0.6, demand_max=0.4),
    dict(demand_max=1.5),
    dict(duration_min=0),
    dict(duration_min=10, duration_max=5),
    dict(arrival_rate=-1.0),
    dict(queue_capacity=0),
])
def test_invalid_workload_config_is_rejected(bad):
    with pytest.raises(ValueError):
        WorkloadConfig(**bad)
