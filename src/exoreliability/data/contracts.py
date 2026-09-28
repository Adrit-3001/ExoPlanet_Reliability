"""Typed records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Kepler Barycentric Julian Date: BKJD = BJD - 2454833.0 (FITS BJDREFI + BJDREFF).
BKJD_OFFSET = 2454833.0


@dataclass(frozen=True)
class LightCurveData:
    """Time-ordered flux samples for one target.

    ``time`` is in BKJD days. ``quality`` holds the Kepler ``SAP_QUALITY`` bit flags
    (zeros when unknown). ``quarter`` labels the Kepler quarter each cadence came from.
    """

    time: FloatArray
    flux: FloatArray
    flux_err: FloatArray
    quality: IntArray
    quarter: IntArray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.time)
        for name in ("flux", "flux_err", "quality", "quarter"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"LightCurveData.{name} length {len(getattr(self, name))} != {n}")

    def __len__(self) -> int:
        return len(self.time)

    def subset(self, mask: NDArray[np.bool_]) -> LightCurveData:
        return LightCurveData(
            time=self.time[mask],
            flux=self.flux[mask],
            flux_err=self.flux_err[mask],
            quality=self.quality[mask],
            quarter=self.quarter[mask],
            meta=dict(self.meta),
        )

    @staticmethod
    def concatenate(parts: list[LightCurveData]) -> LightCurveData:
        if not parts:
            raise ValueError("cannot concatenate zero light curves")
        order = np.argsort(np.concatenate([p.time for p in parts]), kind="stable")
        return LightCurveData(
            time=np.concatenate([p.time for p in parts])[order],
            flux=np.concatenate([p.flux for p in parts])[order],
            flux_err=np.concatenate([p.flux_err for p in parts])[order],
            quality=np.concatenate([p.quality for p in parts])[order],
            quarter=np.concatenate([p.quarter for p in parts])[order],
            meta={},
        )


@dataclass(frozen=True)
class Ephemeris:
    """Linear transit ephemeris. ``epoch_bkjd`` is a mid-transit time in BKJD."""

    period_days: float
    epoch_bkjd: float
    duration_hours: float | None = None
