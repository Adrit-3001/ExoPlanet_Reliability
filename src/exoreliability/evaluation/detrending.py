"""Synthetic benchmark for transit preservation by detrending methods.

Every light curve here is SYNTHETIC: a trapezoidal transit train multiplied by a known
quasi-periodic "spot" trend, plus white noise, on a Kepler-like 29.4-min cadence with two
segments. Because the true trend is known, the detrending error on the transit depth is
measured without noise: ``depth_before`` is measured on ``flux / true_trend`` and
``depth_after`` on ``flux / estimated_trend`` for the *same* noise realisation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import Ephemeris, FloatArray
from exoreliability.preprocessing.detrend import robust_spline_trend, running_median_trend
from exoreliability.preprocessing.phase_fold import in_transit_mask, phase_fold

CADENCE_DAYS = 0.0204335


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    variability_ppm: float  # semi-amplitude of the rotational signal
    rotation_days: float
    duration_hours: float = 3.0
    depth_ppm: float = 1000.0
    period_days: float = 3.7
    noise_ppm: float = 100.0
    gaps_near_transits: bool = False


# Variability levels are anchored to the Kepler field: rotation periods ~1-45 d (median
# ~15 d, McQuillan et al. 2014) and amplitudes from ~100 ppm to ~1 %. "m1_regression" is the
# configuration that exposed the running-median depth loss in Milestone 1. "extreme_*" is
# deliberately beyond what a >=0.5-d-knot spline can follow; it documents the failure
# regime and is not part of the acceptance criterion.
CASES: tuple[SyntheticCase, ...] = (
    SyntheticCase("quiet", 0.0, 15.0),
    SyntheticCase("low_variability", 500.0, 15.0),
    SyntheticCase("moderate_variability", 3000.0, 8.0),
    SyntheticCase("m1_regression", 10000.0, 20.0),
    SyntheticCase("strong_variability", 10000.0, 5.0),
    SyntheticCase("strong_long_transit", 10000.0, 5.0, duration_hours=12.0, period_days=11.3),
    SyntheticCase("strong_gaps_near_transits", 10000.0, 5.0, gaps_near_transits=True),
    SyntheticCase("extreme_fast_rotator", 10000.0, 2.0),
)
ACCEPTANCE_CASES = tuple(c.name for c in CASES if not c.name.startswith("extreme"))


@dataclass
class SyntheticLightCurve:
    case: SyntheticCase
    time: FloatArray
    flux: FloatArray
    true_trend: FloatArray
    ephemeris: Ephemeris


def trapezoid_transit(
    time: FloatArray, eph: Ephemeris, depth: float, ingress_fraction: float = 0.1
) -> FloatArray:
    """Relative flux (1 out of transit) of a symmetric trapezoid with flat bottom at 1 - depth."""
    assert eph.duration_hours is not None
    half = 0.5 * eph.duration_hours / 24.0
    ingress = ingress_fraction * 2 * half
    x = np.abs(phase_fold(time, eph))
    dip = np.clip((half - x) / ingress, 0.0, 1.0)
    return 1.0 - depth * dip


def make_case(case: SyntheticCase, seed: int = 0) -> SyntheticLightCurve:
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.arange(0.0, 85.0, CADENCE_DAYS), np.arange(90.0, 175.0, CADENCE_DAYS)])
    eph = Ephemeris(case.period_days, 1.37 + rng.uniform(0, 0.2), case.duration_hours)
    if case.gaps_near_transits:
        # Remove 0.4 d of data starting 0.05 d after egress for every other transit, and
        # 0.4 d ending 0.05 d before ingress for the others: the trend must be estimated
        # from one side only right next to the transit.
        half = 0.5 * case.duration_hours / 24.0
        n = np.round((t - eph.epoch_bkjd) / eph.period_days)
        dt = t - (eph.epoch_bkjd + n * eph.period_days)
        after = (n % 2 == 0) & (dt > half + 0.05) & (dt < half + 0.45)
        before = (n % 2 == 1) & (dt < -half - 0.05) & (dt > -half - 0.45)
        t = t[~(after | before)]
    a = case.variability_ppm * 1e-6
    phase_mod = 1.0 + 0.3 * np.sin(2 * np.pi * t / 41.0)
    trend = 1.0 + a * phase_mod * (
        np.sin(2 * np.pi * t / case.rotation_days)
        + 0.5 * np.sin(4 * np.pi * t / case.rotation_days + 0.7)
    )
    transit = trapezoid_transit(t, eph, case.depth_ppm * 1e-6)
    flux = transit * trend * (1.0 + rng.normal(0.0, case.noise_ppm * 1e-6, t.size))
    return SyntheticLightCurve(case, t, flux, trend, eph)


TrendFn = Callable[[FloatArray, FloatArray, NDArray[np.bool_] | None], FloatArray]


def method_registry(
    window_days: float = 1.5, knot_spacing_days: float = 0.3
) -> dict[str, tuple[TrendFn, bool]]:
    """name -> (trend function, uses transit mask)."""

    def rm(t: FloatArray, f: FloatArray, ex: NDArray[np.bool_] | None) -> FloatArray:
        return running_median_trend(t, f, window_days=window_days, exclude=ex)

    def sp(t: FloatArray, f: FloatArray, ex: NDArray[np.bool_] | None) -> FloatArray:
        return robust_spline_trend(t, f, knot_spacing_days=knot_spacing_days, exclude=ex)

    return {
        "running_median_blind": (rm, False),
        "running_median_masked": (rm, True),
        "robust_spline_blind": (sp, False),
        "robust_spline_masked": (sp, True),
    }


@dataclass(frozen=True)
class DepthResult:
    case: str
    method: str
    seed: int
    depth_injected_ppm: float
    depth_before_ppm: float
    depth_after_ppm: float
    relative_depth_error: float
    trend_error_rms_ppm: float  # out-of-transit rms of (true-trend-flattened − estimated-flattened)

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def flat_bottom_depth(time: FloatArray, flat: FloatArray, eph: Ephemeris) -> float:
    """Mean depth over the central 50 % of the transit (flat bottom of the trapezoid)."""
    inner = in_transit_mask(time, eph, duration_factor=0.5)
    return float(1.0 - np.mean(flat[inner]))


def evaluate(
    lc: SyntheticLightCurve, method: str, trend_fn: TrendFn, masked: bool, mask_factor: float = 2.0
) -> tuple[DepthResult, FloatArray]:
    exclude = (
        in_transit_mask(lc.time, lc.ephemeris, duration_factor=mask_factor) if masked else None
    )
    est = trend_fn(lc.time, lc.flux, exclude)
    before = flat_bottom_depth(lc.time, lc.flux / lc.true_trend, lc.ephemeris)
    after = flat_bottom_depth(lc.time, lc.flux / est, lc.ephemeris)
    oot = ~in_transit_mask(lc.time, lc.ephemeris, duration_factor=1.2)
    err = (lc.flux / est - lc.flux / lc.true_trend)[oot]
    result = DepthResult(
        case=lc.case.name,
        method=method,
        seed=-1,
        depth_injected_ppm=lc.case.depth_ppm,
        depth_before_ppm=before * 1e6,
        depth_after_ppm=after * 1e6,
        relative_depth_error=(after - before) / before,
        trend_error_rms_ppm=float(np.sqrt(np.mean(err**2)) * 1e6),
    )
    return result, est


def run_benchmark(
    cases: tuple[SyntheticCase, ...] = CASES,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    **method_kwargs: float,
) -> list[DepthResult]:
    results = []
    for case in cases:
        for seed in seeds:
            lc = make_case(case, seed)
            for name, (fn, masked) in method_registry(**method_kwargs).items():
                r, _ = evaluate(lc, name, fn, masked)
                results.append(DepthResult(**{**r.to_dict(), "seed": seed}))  # type: ignore[arg-type]
    return results
