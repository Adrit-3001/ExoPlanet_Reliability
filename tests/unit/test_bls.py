from __future__ import annotations

import numpy as np
import pytest

from exoreliability.baselines.bls import (
    compare_periods,
    downsample_periodogram,
    period_grid,
    run_bls,
)
from exoreliability.config import BLSConfig
from tests.conftest import make_transit_lc

FAST_BLS = BLSConfig(
    min_period_days=1.0,
    max_period_days=10.0,
    durations_hours=[2.0, 3.0, 4.0],
    oversample=2.0,
    n_jobs=1,
)


def test_bls_recovers_injected_period_on_synthetic_data():
    lc = make_transit_lc(n_days=60, period=3.7, depth=1e-3, duration_days=0.125, noise=3e-4, seed=3)
    result, pg = run_bls(lc, FAST_BLS)
    assert result.period_days == pytest.approx(3.7, rel=2e-3)
    assert result.depth == pytest.approx(1e-3, rel=0.25)
    assert result.depth_snr > 10
    assert result.sde > 5
    assert 1.0 <= result.duration_hours <= 4.0
    assert pg["period"].shape == pg["power"].shape == (result.n_periods,)
    assert result.top_peaks[0].period_days == result.period_days


def test_bls_parallel_matches_serial():
    lc = make_transit_lc(n_days=30, period=2.5, seed=4)
    serial, _ = run_bls(lc, FAST_BLS)
    parallel, _ = run_bls(lc, FAST_BLS.model_copy(update={"n_jobs": 2}))
    assert parallel.period_days == serial.period_days
    assert parallel.power == pytest.approx(serial.power)


def test_bls_limits_max_period_by_baseline():
    lc = make_transit_lc(n_days=12, period=2.0)
    result, _ = run_bls(lc, FAST_BLS)
    assert result.max_period_days <= result.baseline_days / 2 + 1e-9


def test_bls_rejects_too_few_points():
    lc = make_transit_lc(n_days=1.0)
    with pytest.raises(ValueError):
        run_bls(lc, FAST_BLS)


def test_bls_rejects_oversized_grid():
    lc = make_transit_lc(n_days=60)
    with pytest.raises(ValueError, match="max_grid_size"):
        run_bls(lc, FAST_BLS.model_copy(update={"max_grid_size": 100}))


def test_bls_config_rejects_duration_longer_than_min_period():
    with pytest.raises(ValueError):
        BLSConfig(min_period_days=0.2, durations_hours=[6.0])


def test_period_grid_spacing():
    grid = period_grid(
        baseline_days=100, min_period=1, max_period=10, min_duration_days=0.1, oversample=2
    )
    steps = np.diff(np.log(grid))
    assert grid[0] == pytest.approx(1) and grid[-1] == pytest.approx(10)
    assert steps.max() <= 0.1 / (2 * 100) + 1e-12


@pytest.mark.parametrize(
    ("bls", "cat", "relation", "harmonic"),
    [
        (10.0, 10.0, "match", "1/1"),
        (10.005, 10.0, "match", "1/1"),
        (20.0, 10.0, "harmonic", "2/1"),
        (5.0, 10.0, "harmonic", "1/2"),
        (15.0, 10.0, "harmonic", "3/2"),
        (13.7, 10.0, "none", None),
    ],
)
def test_compare_periods(bls, cat, relation, harmonic):
    out = compare_periods(bls, cat)
    assert out["relation"] == relation
    assert out["harmonic"] == harmonic


def test_downsample_periodogram_preserves_peak():
    period = np.linspace(1, 10, 10_000)
    power = np.zeros_like(period)
    power[1234] = 5.0
    out = downsample_periodogram(period, power, max_points=100)
    assert len(out["period"]) == 100
    assert max(out["power"]) == 5.0
    assert period[1234] in out["period"]
