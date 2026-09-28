"""Removal of slow stellar/instrumental trends."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import FloatArray


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
