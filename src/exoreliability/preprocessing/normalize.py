"""Flux normalization."""

from __future__ import annotations

import numpy as np

from exoreliability.data.contracts import LightCurveData


def normalize_per_quarter(lc: LightCurveData) -> tuple[LightCurveData, dict[int, float]]:
    """Divide flux and uncertainty by the median flux of each Kepler quarter.

    Kepler targets fall on a different CCD module each quarter (the spacecraft rolls), so
    absolute flux levels differ between quarters; normalising each quarter to a median of 1
    puts them on a common relative scale. Statistics are computed from this target's own
    data only, so no information is shared across targets or dataset splits.

    Returns the normalized light curve and the median used for each quarter.
    """
    flux = lc.flux.copy()
    flux_err = lc.flux_err.copy()
    medians: dict[int, float] = {}
    for q in np.unique(lc.quarter):
        sel = lc.quarter == q
        med = float(np.median(lc.flux[sel]))
        if not np.isfinite(med) or med <= 0:
            raise ValueError(
                f"quarter {q}: median flux {med} is not positive; clean the data first"
            )
        flux[sel] /= med
        flux_err[sel] /= med
        medians[int(q)] = med
    return (
        LightCurveData(lc.time, flux, flux_err, lc.quality, lc.quarter, dict(lc.meta)),
        medians,
    )
