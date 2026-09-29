"""Transit-preserving detrending on SYNTHETIC light curves (evaluation.detrending)."""

from __future__ import annotations

import numpy as np
import pytest

from exoreliability.config import DetrendConfig, PreprocessingConfig
from exoreliability.data.contracts import Ephemeris
from exoreliability.evaluation.detrending import (
    ACCEPTANCE_CASES,
    CASES,
    evaluate,
    make_case,
    method_registry,
)
from exoreliability.preprocessing.detrend import _spline_knots, robust_spline_trend
from exoreliability.preprocessing.phase_fold import in_transit_mask
from exoreliability.preprocessing.pipeline import preprocess_files
from tests.conftest import make_transit_lc, write_fake_kepler_fits

CASE = {c.name: c for c in CASES}
METHODS = method_registry()

# Acceptance thresholds (docs/data_contract.md §6): systematic relative depth error (mean
# over noise realisations) <= 2 %, random error (rms over realisations) <= 5 %.
MEAN_TOL, RMS_TOL = 0.02, 0.05


@pytest.mark.parametrize("case", ACCEPTANCE_CASES)
def test_masked_spline_preserves_depth(case):
    fn, masked = METHODS["robust_spline_masked"]
    errs = np.array(
        [
            evaluate(make_case(CASE[case], seed), "robust_spline_masked", fn, masked)[
                0
            ].relative_depth_error
            for seed in range(6)
        ]
    )
    assert abs(errs.mean()) <= MEAN_TOL
    assert np.sqrt(np.mean(errs**2)) <= RMS_TOL


def test_old_running_median_loses_depth_on_variable_star():
    """Characterises the Milestone 1 method: ~30 % depth loss on the regression case."""
    fn, masked = METHODS["running_median_blind"]
    r, _ = evaluate(make_case(CASE["m1_regression"], 0), "running_median_blind", fn, masked)
    assert r.relative_depth_error < -0.2


@pytest.mark.parametrize("case", ["moderate_variability", "strong_variability"])
def test_variability_removed(case):
    fn, masked = METHODS["robust_spline_masked"]
    lc = make_case(CASE[case], 0)
    r, _ = evaluate(lc, "robust_spline_masked", fn, masked)
    raw_rms = np.std(lc.true_trend) * 1e6
    assert raw_rms > 1000  # the case really is variable
    assert r.trend_error_rms_ppm < 0.5 * CASE[case].noise_ppm


def test_deterministic():
    lc = make_case(CASE["strong_gaps_near_transits"], 3)
    ex = in_transit_mask(lc.time, lc.ephemeris, 2.0)
    a = robust_spline_trend(lc.time, lc.flux, knot_spacing_days=0.3, exclude=ex)
    b = robust_spline_trend(lc.time, lc.flux, knot_spacing_days=0.3, exclude=ex)
    np.testing.assert_array_equal(a, b)


def test_segments_are_not_bridged():
    t = np.concatenate([np.arange(0, 5, 0.02), np.arange(10, 15, 0.02)])
    f = np.where(t < 7, 1.0, 1.01)
    trend = robust_spline_trend(t, f, knot_spacing_days=0.3, gap_days=1.0)
    np.testing.assert_allclose(trend[t < 7], 1.0, atol=1e-9)
    np.testing.assert_allclose(trend[t > 7], 1.01, atol=1e-9)


def test_masked_points_do_not_enter_fit():
    t = np.arange(0, 10, 0.02)
    f = np.ones_like(t)
    hole = (t > 4.8) & (t < 5.3)
    f[hole] = 0.5  # huge dip, masked
    trend = robust_spline_trend(t, f, knot_spacing_days=0.3, exclude=hole)
    np.testing.assert_allclose(trend, 1.0, atol=1e-9)


def test_knots_avoid_data_holes():
    t = np.concatenate([np.arange(0, 2, 0.02), np.arange(3, 5, 0.02)])  # 1-day hole
    knots = _spline_knots(t, 0.0, 5.0, 0.3, 10)
    assert not np.any((knots > 2.15) & (knots < 2.85))
    assert len(knots) > 5


def test_single_large_outlier_is_not_followed():
    lc = make_transit_lc(n_days=20, noise=1e-4, depth=0.0)
    f = lc.flux.copy()
    f[400:403] += 0.05  # flare-like spike
    trend = robust_spline_trend(lc.time, f, knot_spacing_days=0.3)
    assert np.max(np.abs(trend - 1.0)) < 5e-4


def test_pipeline_uses_spline_with_all_koi_masks(tmp_path):
    lc = make_transit_lc(n_days=30, period=3.7, epoch=1.3, duration_days=0.125)
    path = write_fake_kepler_fits(
        tmp_path / "q1.fits",
        kepid=1,
        quarter=1,
        time=lc.time,
        flux=lc.flux * 1e4,
        flux_err=lc.flux_err * 1e4,
    )
    cfg = PreprocessingConfig(
        detrend=DetrendConfig(method="robust_spline", mask_known_transits=True)
    )
    ephs = [Ephemeris(3.7, 131.3, 3.0), Ephemeris(7.1, 133.0, 5.0)]
    out = preprocess_files([path], cfg, ephemerides=ephs)
    mask = out.frame["in_known_transit_mask"].to_numpy()
    t = out.frame["time_bkjd"].to_numpy()
    expected = in_transit_mask(t, ephs[0], 2.0) | in_transit_mask(t, ephs[1], 2.0)
    np.testing.assert_array_equal(mask, expected)
    assert out.record["config"]["detrend"]["method"] == "robust_spline"
