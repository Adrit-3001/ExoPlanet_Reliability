"""Removal of slow stellar/instrumental trends."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LSQUnivariateSpline

from exoreliability.data.contracts import FloatArray
from exoreliability.preprocessing.clean import robust_std


def contiguous_segments(time: FloatArray, gap_days: float) -> list[slice]:
    """Split a sorted time array wherever consecutive samples are more than ``gap_days`` apart."""
    if len(time) == 0:
        return []
    breaks = np.flatnonzero(np.diff(time) > gap_days) + 1
    edges = np.concatenate([[0], breaks, [len(time)]])
    return [slice(int(a), int(b)) for a, b in pairwise(edges)]


def running_median_trend(
    time: FloatArray,
    flux: FloatArray,
    *,
    window_days: float,
    gap_days: float = 0.5,
    min_points: int = 10,
    exclude: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Estimate a slowly varying trend with a time-windowed running median.

    For each sample the trend is the median of flux within ``±window_days/2``, using only
    samples in the same contiguous segment (no smoothing across data gaps) and not marked in
    ``exclude`` (e.g. known in-transit points). Where fewer than ``min_points`` samples are
    available the trend is linearly interpolated from neighbouring estimates in the segment.

    A running median suppresses features much shorter than the window (such as transits)
    but will partially absorb transits whose duration approaches the window, so the window
    must be chosen several times longer than the longest transit of interest.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    if np.any(np.diff(time) < 0):
        raise ValueError("time must be sorted")
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    usable = np.ones_like(time, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)

    trend = np.full_like(flux, np.nan)
    half = window_days / 2.0
    for seg in contiguous_segments(time, gap_days):
        t, f, u = time[seg], flux[seg], usable[seg]
        tu, fu = t[u], f[u]
        seg_trend = np.full_like(f, np.nan)
        if tu.size:
            lo = np.searchsorted(tu, t - half, side="left")
            hi = np.searchsorted(tu, t + half, side="right")
            for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
                if b - a >= min_points:
                    seg_trend[i] = np.median(fu[a:b])
        good = np.isfinite(seg_trend)
        if good.any():
            if not good.all():
                seg_trend[~good] = np.interp(t[~good], t[good], seg_trend[good])
        else:
            # Segment too short for the window: fall back to its (usable) median level.
            seg_trend[:] = np.median(fu) if fu.size else np.median(f)
        trend[seg] = seg_trend
    return trend


def _spline_knots(
    t_fit: FloatArray, start: float, end: float, spacing: float, min_points: int
) -> FloatArray:
    """Interior knots every ``spacing`` days.

    A candidate knot is dropped if it would leave fewer than ``min_points`` fit samples in
    its knot interval, or if fewer than ``min_points`` fit samples lie within
    ``±spacing/2`` of it. The second rule keeps knots out of data holes (masked transits,
    short gaps): a knot inside a hole leaves the local cubic's curvature unconstrained by
    data and makes the fit oscillate there.
    """
    kept: list[float] = []
    last = start
    half = 0.5 * spacing
    for c in np.arange(start + spacing, end - 0.5 * spacing, spacing):
        n = np.searchsorted(t_fit, c, side="right") - np.searchsorted(t_fit, last, side="right")
        local = np.searchsorted(t_fit, c + half, side="right") - np.searchsorted(
            t_fit, c - half, side="left"
        )
        if n >= min_points and local >= min_points:
            kept.append(float(c))
            last = c
    while kept and t_fit.size - np.searchsorted(t_fit, kept[-1], side="right") < min_points:
        kept.pop()
    return np.asarray(kept)


def _fit_segment_spline(
    t: FloatArray, f: FloatArray, fit: NDArray[np.bool_], spacing: float, min_points: int
) -> FloatArray:
    """Least-squares cubic spline through ``f[fit]`` evaluated at every ``t``.

    Falls back to coarser knots, then to a low-order polynomial, when there is too little
    data for the requested knot spacing.
    """
    tf, ff = t[fit], f[fit]
    if tf.size < 4:
        return np.full_like(f, np.median(ff) if ff.size else np.median(f))
    for factor in (1, 2, 4):
        knots = _spline_knots(tf, float(t[0]), float(t[-1]), spacing * factor, min_points)
        try:
            spline = LSQUnivariateSpline(tf, ff, knots, k=3, bbox=[float(t[0]), float(t[-1])])
        except ValueError:
            continue
        return np.asarray(spline(t), dtype=np.float64)
    deg = min(2, tf.size - 1)
    coeffs = np.polynomial.polynomial.polyfit(tf - t[0], ff, deg)
    return np.asarray(np.polynomial.polynomial.polyval(t - t[0], coeffs), dtype=np.float64)


def robust_spline_trend(
    time: FloatArray,
    flux: FloatArray,
    *,
    knot_spacing_days: float,
    gap_days: float = 0.5,
    min_points: int = 10,
    sigma_lower: float = 3.0,
    sigma_upper: float = 3.0,
    max_iter: int = 3,
    exclude: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Trend from an iteratively sigma-clipped least-squares cubic B-spline.

    Fitted independently on each contiguous segment (no bridging of data gaps). Samples in
    ``exclude`` (known transit windows) never enter the fit, but the trend is evaluated
    there, so masked transits are divided by an interpolated baseline rather than by a
    trend that has partly followed the dip. ``max_iter`` counts fits: after each fit except
    the last, samples whose residual deviates from the local (``knot_spacing_days``-wide)
    running median of residuals by less than ``-sigma_lower`` or more than ``+sigma_upper``
    robust standard deviations are dropped from the next fit (flares, spikes, unmasked
    short dips). Using the local median keeps smooth model misfit from being clipped.

    Unlike a running median/biweight (location estimators), a cubic spline follows the
    local slope and curvature of stellar variability, so transits falling on a steep
    gradient are not partially absorbed. ``knot_spacing_days`` sets the shortest
    variability timescale followed; it must be long compared with unmasked transit
    durations when used blind.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    if np.any(np.diff(time) < 0):
        raise ValueError("time must be sorted")
    if knot_spacing_days <= 0:
        raise ValueError("knot_spacing_days must be positive")
    usable = np.isfinite(flux)
    if exclude is not None:
        usable &= ~np.asarray(exclude, dtype=bool)

    trend = np.full_like(flux, np.nan)
    for seg in contiguous_segments(time, gap_days):
        t, f, u = time[seg], flux[seg], usable[seg]
        fit = u.copy()
        seg_trend = _fit_segment_spline(t, f, fit, knot_spacing_days, min_points)
        for _ in range(max_iter - 1):
            # Outliers are judged against the *local* median of the residuals so that
            # smooth misfit (e.g. next to a masked window) is not clipped; clipping it
            # would enlarge the hole and destabilise the next fit.
            resid = f - seg_trend
            local = running_median_trend(
                t,
                resid,
                window_days=knot_spacing_days,
                gap_days=gap_days,
                min_points=min_points,
                exclude=~fit,
            )
            dev = resid - local
            sigma = robust_std(dev[fit]) if fit.any() else 0.0
            if not np.isfinite(sigma) or sigma == 0:
                break
            new_fit = u & (dev > -sigma_lower * sigma) & (dev < sigma_upper * sigma)
            if np.array_equal(new_fit, fit) or new_fit.sum() < 4:
                break
            fit = new_fit
            seg_trend = _fit_segment_spline(t, f, fit, knot_spacing_days, min_points)
        trend[seg] = seg_trend
    return trend
