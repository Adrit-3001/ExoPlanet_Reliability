"""Removal of invalid and flagged cadences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import FloatArray, IntArray, LightCurveData

MAD_TO_SIGMA = 1.4826


def finite_mask(lc: LightCurveData) -> NDArray[np.bool_]:
    """True where time, flux and flux uncertainty are all finite."""
    return np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)


def quality_mask(quality: IntArray, bitmask: int) -> NDArray[np.bool_]:
    """True where none of the bits in ``bitmask`` are set in the Kepler quality flags."""
    return (np.asarray(quality, dtype=np.int64) & int(bitmask)) == 0


def clean(lc: LightCurveData, quality_bitmask: int) -> tuple[LightCurveData, dict[str, int]]:
    """Drop non-finite samples and cadences flagged by ``quality_bitmask``.

    Returns the cleaned light curve and counts of removed samples per reason.
    A sample failing both checks is counted under ``non_finite``.
    """
    finite = finite_mask(lc)
    good_quality = quality_mask(lc.quality, quality_bitmask)
    keep = finite & good_quality
    stats = {
        "input": len(lc),
        "non_finite": int((~finite).sum()),
        "quality_flagged": int((finite & ~good_quality).sum()),
        "kept": int(keep.sum()),
    }
    return lc.subset(keep), stats


def robust_std(values: FloatArray) -> float:
    """Gaussian-equivalent standard deviation from the median absolute deviation."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def upper_outlier_mask(flux: FloatArray, sigma: float) -> NDArray[np.bool_]:
    """True for samples more than ``sigma`` robust standard deviations *above* the median.

    Only positive outliers (e.g. cosmic rays) are flagged, so transit dips are never clipped.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    med = np.nanmedian(flux)
    return np.asarray(flux > med + sigma * robust_std(flux))
