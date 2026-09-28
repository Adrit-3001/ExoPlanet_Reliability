from __future__ import annotations

import numpy as np
import pytest

from exoreliability.perturbations.gaussian_noise import GaussianNoise, point_to_point_sigma
from tests.conftest import make_transit_lc


@pytest.fixture
def lc():
    return make_transit_lc(n_days=40, noise=2e-4, seed=11)


def test_severity_zero_is_identity_and_consumes_no_randomness(lc):
    rng = np.random.default_rng(5)
    out = GaussianNoise().apply(lc, 0.0, rng)
    np.testing.assert_array_equal(out.light_curve.flux, lc.flux)
    np.testing.assert_array_equal(out.light_curve.flux_err, lc.flux_err)
    assert out.light_curve.flux is not lc.flux  # a copy, not an alias
    assert out.params["sigma_added"] == 0.0
    assert rng.random() == np.random.default_rng(5).random()


def test_fixed_seed_is_reproducible(lc):
    a = GaussianNoise().apply(lc, 1.0, np.random.default_rng(42)).light_curve.flux
    b = GaussianNoise().apply(lc, 1.0, np.random.default_rng(42)).light_curve.flux
    c = GaussianNoise().apply(lc, 1.0, np.random.default_rng(43)).light_curve.flux
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shape_and_times_preserved(lc):
    out = GaussianNoise().apply(lc, 2.0, np.random.default_rng(0)).light_curve
    assert len(out) == len(lc)
    np.testing.assert_array_equal(out.time, lc.time)


@pytest.mark.parametrize("severity", [0.5, 1.0, 2.0])
def test_added_noise_matches_severity_semantics(lc, severity):
    out = GaussianNoise().apply(lc, severity, np.random.default_rng(1))
    sigma_ref = point_to_point_sigma(lc.flux)
    assert out.params["sigma_ref"] == pytest.approx(sigma_ref)
    added = out.light_curve.flux - lc.flux
    assert np.std(added) == pytest.approx(severity * sigma_ref, rel=0.03)
    assert abs(np.mean(added)) < 5 * severity * sigma_ref / np.sqrt(added.size)


def test_sigma_ref_estimates_white_noise_level(lc):
    # 2e-4 white noise; the few in-transit points do not bias the estimator much.
    assert point_to_point_sigma(lc.flux) == pytest.approx(2e-4, rel=0.05)


def test_metadata_sigma_ref_overrides_estimate(lc):
    out = GaussianNoise().apply(lc, 1.0, np.random.default_rng(0), metadata={"sigma_ref": 1e-3})
    assert out.params["sigma_ref"] == 1e-3 and out.params["sigma_ref_source"] == "metadata"


def test_flux_err_inflated_in_quadrature(lc):
    out = GaussianNoise().apply(lc, 1.0, np.random.default_rng(0), metadata={"sigma_ref": 3e-4})
    np.testing.assert_allclose(out.light_curve.flux_err, np.sqrt(lc.flux_err**2 + 9e-8))
    kept = GaussianNoise(update_flux_err=False).apply(lc, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(kept.light_curve.flux_err, lc.flux_err)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_severity_rejected(lc, bad):
    with pytest.raises(ValueError):
        GaussianNoise().apply(lc, bad, np.random.default_rng(0))


def test_input_not_mutated(lc):
    before = lc.flux.copy()
    GaussianNoise().apply(lc, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(lc.flux, before)
