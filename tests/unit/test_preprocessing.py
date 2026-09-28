from __future__ import annotations

import numpy as np
import pytest

from exoreliability.config import DetrendConfig, PreprocessingConfig
from exoreliability.data.contracts import Ephemeris, LightCurveData
from exoreliability.data.mast import read_kepler_fits
from exoreliability.preprocessing.clean import clean, quality_mask, robust_std, upper_outlier_mask
from exoreliability.preprocessing.detrend import contiguous_segments, running_median_trend
from exoreliability.preprocessing.normalize import normalize_per_quarter
from exoreliability.preprocessing.phase_fold import in_transit_mask, phase_fold
from exoreliability.preprocessing.pipeline import load_processed, preprocess_files, save_processed
from exoreliability.preprocessing.resample import bin_curve
from tests.conftest import make_transit_lc, write_fake_kepler_fits


def _lc(time, flux, flux_err=None, quality=None, quarter=None) -> LightCurveData:
    n = len(time)
    return LightCurveData(
        time=np.asarray(time, float),
        flux=np.asarray(flux, float),
        flux_err=np.ones(n) if flux_err is None else np.asarray(flux_err, float),
        quality=np.zeros(n, np.int64) if quality is None else np.asarray(quality, np.int64),
        quarter=np.ones(n, np.int64) if quarter is None else np.asarray(quarter, np.int64),
    )


# -- cleaning -------------------------------------------------------------------


def test_clean_removes_nan_inf_and_flagged():
    lc = _lc(
        time=[0, 1, 2, np.nan, 4, 5],
        flux=[1, np.nan, 1, 1, np.inf, 1],
        quality=[0, 0, 0, 0, 0, 0b100],
    )
    out, stats = clean(lc, quality_bitmask=0b100)
    np.testing.assert_array_equal(out.time, [0, 2])
    assert stats == {"input": 6, "non_finite": 3, "quality_flagged": 1, "kept": 2}


def test_clean_nan_flux_err_is_removed():
    lc = _lc([0, 1, 2], [1, 1, 1], flux_err=[1, np.nan, 1])
    out, _ = clean(lc, quality_bitmask=0)
    assert len(out) == 2


def test_quality_mask_only_checks_selected_bits():
    q = np.array([0, 1, 2, 3, 8])
    np.testing.assert_array_equal(quality_mask(q, bitmask=0b010), [True, True, False, False, True])


def test_upper_outlier_mask_never_flags_dips():
    flux = np.ones(1000) + np.random.default_rng(0).normal(0, 1e-4, 1000)
    flux[100] = 1.01  # cosmic ray
    flux[200] = 0.99  # deep dip
    mask = upper_outlier_mask(flux, sigma=5)
    assert mask[100] and not mask[200]
    assert mask.sum() == 1


def test_robust_std_gaussian():
    x = np.random.default_rng(1).normal(0, 2.0, 200_000)
    assert robust_std(x) == pytest.approx(2.0, rel=0.02)


# -- normalization ----------------------------------------------------------------


def test_normalize_per_quarter_gives_unit_median_per_quarter():
    lc = _lc(
        time=np.arange(6),
        flux=[100, 110, 90, 1000, 1100, 900],
        flux_err=[10, 10, 10, 10, 10, 10],
        quarter=[1, 1, 1, 2, 2, 2],
    )
    out, medians = normalize_per_quarter(lc)
    assert medians == {1: 100.0, 2: 1000.0}
    for q in (1, 2):
        assert np.median(out.flux[out.quarter == q]) == pytest.approx(1.0)
    np.testing.assert_allclose(out.flux_err, [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])


def test_normalize_rejects_non_positive_median():
    with pytest.raises(ValueError):
        normalize_per_quarter(_lc([0, 1, 2], [-1, -1, -1]))


# -- detrending -------------------------------------------------------------------


def test_contiguous_segments_split_on_gaps():
    t = np.array([0, 0.1, 0.2, 5.0, 5.1])
    assert contiguous_segments(t, gap_days=1.0) == [slice(0, 3), slice(3, 5)]


EPH_3D = Ephemeris(3.0, 131.3, 0.1 * 24)


def test_running_median_preserves_transit_depth_on_flat_baseline():
    lc = make_transit_lc(noise=5e-5, depth=1e-3, duration_days=0.1, period=3.0)
    detrended = lc.flux / running_median_trend(lc.time, lc.flux, window_days=1.0)
    in_tr = in_transit_mask(lc.time, EPH_3D, duration_factor=0.8)
    assert 1 - np.median(detrended[in_tr]) == pytest.approx(1e-3, rel=0.05)


def test_running_median_removes_slow_trend_but_absorbs_depth_on_steep_gradients():
    """Characterises a known limitation (docs/progress.md, Known Issues).

    When trend change across the window (here ~3e-3) greatly exceeds the noise, in-transit
    points are removed from the middle of the window's value distribution, so the median
    shifts upward and part of the transit depth is absorbed (~20% here).
    """
    lc = make_transit_lc(noise=5e-5, depth=1e-3, duration_days=0.1, period=3.0)
    flux = lc.flux * (1.0 + 0.01 * np.sin(2 * np.pi * lc.time / 20.0))
    detrended = flux / running_median_trend(lc.time, flux, window_days=1.0)
    in_tr = in_transit_mask(lc.time, EPH_3D, duration_factor=0.8)
    near_tr = in_transit_mask(lc.time, EPH_3D, duration_factor=1.2)
    interior = (lc.time > lc.time[0] + 0.5) & (lc.time < lc.time[-1] - 0.5)  # see edge test
    assert np.std(detrended[interior & ~near_tr]) < 1e-4  # raw trend std is ~7e-3
    depth = 1 - np.median(detrended[interior & in_tr])
    assert 0.5e-3 < depth < 0.95e-3


def test_running_median_is_biased_at_segment_edges_on_sloped_trends():
    """Documents a known limitation: the window is one-sided at edges, so a gradient biases
    the trend there by ~slope × window/4. Revisit if edge artifacts matter (docs/progress.md)."""
    t = np.arange(0.0, 10.0, 0.02)
    flux = 1.0 + 0.001 * t  # slope 1e-3 / day
    trend = running_median_trend(t, flux, window_days=1.0)
    resid = flux / trend - 1
    assert abs(resid[0]) == pytest.approx(0.001 * 0.25, rel=0.1)
    assert np.abs(resid[(t > 1) & (t < 9)]).max() < 2e-5  # one-sample window asymmetry


def test_running_median_exclude_mask_ignores_masked_points():
    t = np.arange(0, 10, 0.02)
    flux = np.ones_like(t)
    flux[(t > 4.9) & (t < 5.1)] = 0.5
    exclude = (t > 4.8) & (t < 5.2)
    trend = running_median_trend(t, flux, window_days=0.5, exclude=exclude)
    np.testing.assert_allclose(trend, 1.0)


def test_running_median_does_not_bridge_gaps():
    t = np.concatenate([np.arange(0, 2, 0.02), np.arange(10, 12, 0.02)])
    flux = np.where(t < 5, 1.0, 2.0)
    trend = running_median_trend(t, flux, window_days=20.0, gap_days=1.0)
    np.testing.assert_allclose(trend[t < 5], 1.0)
    np.testing.assert_allclose(trend[t > 5], 2.0)


def test_running_median_requires_sorted_time():
    with pytest.raises(ValueError):
        running_median_trend(np.array([1.0, 0.0]), np.array([1.0, 1.0]), window_days=1)


# -- phase folding / binning ---------------------------------------------------------


def test_phase_fold_maps_known_times():
    eph = Ephemeris(period_days=10.0, epoch_bkjd=100.0)
    t = np.array([100.0, 110.0, 90.0, 102.5, 97.5, 105.0, 104.999])
    np.testing.assert_allclose(
        phase_fold(t, eph), [0.0, 0.0, 0.0, 2.5, -2.5, -5.0, 4.999], atol=1e-9
    )


def test_in_transit_mask_width():
    eph = Ephemeris(period_days=10.0, epoch_bkjd=0.0, duration_hours=24.0)
    t = np.array([0.0, 0.7, 0.8, 9.3, 5.0])
    np.testing.assert_array_equal(
        in_transit_mask(t, eph, duration_factor=1.5), [True, True, False, True, False]
    )


def test_bin_curve_shape_and_values():
    x = np.array([0.1, 0.2, 0.6, 0.9, 1.0, 5.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 100.0])
    centres, med, counts = bin_curve(x, y, n_bins=2, x_range=(0.0, 1.0))
    np.testing.assert_allclose(centres, [0.25, 0.75])
    np.testing.assert_allclose(med, [2.0, 7.0])
    np.testing.assert_array_equal(counts, [2, 3])
    _, mean, _ = bin_curve(x, y, n_bins=4, x_range=(0.0, 1.0), statistic="mean")
    assert mean.shape == (4,)
    assert np.isnan(mean[1])


# -- FITS reading + end-to-end pipeline on SYNTHETIC files ---------------------------


def _write_quarters(tmp_path, kepid=123):
    files = []
    for q, level in ((1, 1000.0), (2, 5000.0)):
        lc = make_transit_lc(n_days=30, t_start=130 + 40 * (q - 1), seed=q)
        flux = lc.flux * level
        flux[5] = np.nan
        quality = np.zeros(lc.time.size, np.int32)
        quality[10] = 1  # attitude tweak -> in default bitmask
        files.append(
            write_fake_kepler_fits(
                tmp_path / f"q{q}.fits",
                kepid=kepid,
                quarter=q,
                time=lc.time,
                flux=flux,
                flux_err=np.full(lc.time.size, 0.2 * level),
                quality=quality,
            )
        )
    return files


def test_read_kepler_fits_returns_unfiltered_columns(tmp_path):
    (path, _) = _write_quarters(tmp_path)
    lc = read_kepler_fits(path)
    assert np.isnan(lc.flux[5]) and lc.quality[10] == 1
    assert lc.meta["quarter"] == 1 and lc.meta["data_release"] == 25
    assert (lc.quarter == 1).all()


def test_preprocess_files_end_to_end(tmp_path, tmp_paths):
    files = _write_quarters(tmp_path)
    processed = preprocess_files(files, PreprocessingConfig(detrend=DetrendConfig(window_days=1.0)))
    f = processed.frame
    assert np.isfinite(f.to_numpy()).all()
    assert set(f["quarter"]) == {1, 2}
    for q in (1, 2):  # both quarters on a common scale despite 5x level difference
        assert np.median(f.loc[f["quarter"] == q, "flux_norm"]) == pytest.approx(1.0, abs=1e-3)
    assert (np.diff(f["time_bkjd"]) > 0).all()
    q1 = processed.record["quarters"][0]["clean"]
    assert q1["non_finite"] == 1 and q1["quality_flagged"] == 1
    assert len(processed.record["inputs"]) == 2

    save_processed(processed, tmp_paths, 123)
    loaded = load_processed(tmp_paths, 123)
    assert loaded is not None and len(loaded.frame) == len(f)


def test_mask_known_transits_requires_ephemerides(tmp_path):
    files = _write_quarters(tmp_path)
    cfg = PreprocessingConfig(detrend=DetrendConfig(mask_known_transits=True))
    with pytest.raises(ValueError):
        preprocess_files(files, cfg)
