"""Phase folding with a linear ephemeris."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import Ephemeris, FloatArray


def phase_fold(time: FloatArray, ephemeris: Ephemeris) -> FloatArray:
    """Time from the nearest mid-transit, in days, within ``[-P/2, P/2)``.

    ``time`` and ``ephemeris.epoch_bkjd`` must use the same time system (BKJD for Kepler).
    A sample exactly at an epoch maps to 0; one half a period later maps to ``-P/2``.
    """
    p = ephemeris.period_days
    if p <= 0:
        raise ValueError("period must be positive")
    t = np.asarray(time, dtype=np.float64)
    return np.mod(t - ephemeris.epoch_bkjd + 0.5 * p, p) - 0.5 * p


def in_transit_mask(
    time: FloatArray, ephemeris: Ephemeris, duration_factor: float = 1.5
) -> NDArray[np.bool_]:
    """True within ``duration_factor × duration / 2`` of any predicted mid-transit."""
    if ephemeris.duration_hours is None:
        raise ValueError("ephemeris has no duration")
    half_width_days = 0.5 * duration_factor * ephemeris.duration_hours / 24.0
    return np.abs(phase_fold(time, ephemeris)) < half_width_days
