"""Binning to fixed-length representations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import FloatArray


def bin_curve(
    x: FloatArray,
    y: FloatArray,
    n_bins: int,
    x_range: tuple[float, float],
    statistic: str = "median",
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Bin ``y`` on a uniform grid of ``n_bins`` over ``x_range``.

    Always returns arrays of length ``n_bins``: bin centres, the per-bin statistic
    (``median`` or ``mean``; NaN for empty bins), and the per-bin sample count.
    Samples outside ``x_range`` are ignored; the upper edge is inclusive.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    lo, hi = x_range
    if not hi > lo:
        raise ValueError("x_range must be increasing")
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    inside = (x >= lo) & (x <= hi) & np.isfinite(y)
    idx = np.clip(np.searchsorted(edges, x[inside], side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    values = np.full(n_bins, np.nan)
    if statistic == "mean":
        sums = np.bincount(idx, weights=y[inside], minlength=n_bins)
        nonzero = counts > 0
        values[nonzero] = sums[nonzero] / counts[nonzero]
    else:
        order = np.argsort(idx, kind="stable")
        sorted_idx, sorted_y = idx[order], y[inside][order]
        starts = np.searchsorted(sorted_idx, np.arange(n_bins), side="left")
        stops = np.searchsorted(sorted_idx, np.arange(n_bins), side="right")
        for b in np.flatnonzero(counts):
            values[b] = np.median(sorted_y[starts[b] : stops[b]])
    return centres, values, counts
