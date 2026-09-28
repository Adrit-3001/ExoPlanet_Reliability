"""Box Least Squares transit search — a thin wrapper around ``astropy.timeseries.BoxLeastSquares``.

The search is *blind*: it uses only the light curve and the configured period/duration
grid, never the catalog ephemeris. Catalog periods are compared only after the search
(``compare_periods``), so the comparison is a pipeline check and not a tuned result.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from itertools import pairwise
from typing import Any

import numpy as np
from astropy.timeseries import BoxLeastSquares

from exoreliability.config import BLSConfig
from exoreliability.data.contracts import FloatArray, LightCurveData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BLSPeak:
    period_days: float
    power: float


@dataclass(frozen=True)
class BLSResult:
    """Best BLS peak plus diagnostics. Depths are fractional (1e-3 = 1000 ppm)."""

    period_days: float
    duration_hours: float
    transit_time_bkjd: float
    depth: float
    depth_err: float
    depth_snr: float
    power: float
    log_likelihood: float
    sde: float
    depth_odd: float
    depth_odd_err: float
    depth_even: float
    depth_even_err: float
    harmonic_delta_log_likelihood: float
    n_transits_with_data: int
    objective: str
    n_points: int
    baseline_days: float
    n_periods: int
    min_period_days: float
    max_period_days: float
    durations_hours: list[float]
    top_peaks: list[BLSPeak]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def period_grid(
    baseline_days: float,
    min_period: float,
    max_period: float,
    min_duration_days: float,
    oversample: float,
) -> FloatArray:
    """Log-uniform period grid fine enough not to smear the shortest transit.

    A fractional period error δP/P accumulates a phase drift of (T/P)·δP = T·δP/P over a
    baseline T. Requiring this drift to stay below ``min_duration / oversample`` gives a
    constant step in ln P of ``min_duration / (oversample · T)``.
    """
    if not (0 < min_period < max_period):
        raise ValueError("require 0 < min_period < max_period")
    step = min_duration_days / (oversample * baseline_days)
    n = math.ceil(math.log(max_period / min_period) / step) + 1
    return np.exp(np.linspace(math.log(min_period), math.log(max_period), n))


def _top_peaks(
    periods: FloatArray, power: FloatArray, k: int = 5, sep: float = 0.01
) -> list[BLSPeak]:
    """Up to ``k`` highest peaks whose periods differ from each other by more than ``sep`` (fractional)."""
    peaks: list[BLSPeak] = []
    for i in np.argsort(power)[::-1]:
        p = float(periods[i])
        if all(abs(p - q.period_days) / q.period_days > sep for q in peaks):
            peaks.append(BLSPeak(p, float(power[i])))
            if len(peaks) == k:
                break
    return peaks


_PG_FIELDS = (
    "power",
    "duration",
    "transit_time",
    "depth",
    "depth_err",
    "depth_snr",
    "log_likelihood",
)


def _power_chunk(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    objective: str,
) -> dict[str, FloatArray]:
    pg = BoxLeastSquares(t, y, dy=dy).power(periods, durations, objective=objective)
    return {k: np.asarray(getattr(pg, k), dtype=np.float64) for k in _PG_FIELDS}


def _power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    objective: str,
    n_jobs: int,
) -> dict[str, FloatArray]:
    """Evaluate the BLS periodogram, optionally split across processes by period chunk.

    Each period is evaluated independently by Astropy, so chunking does not change results.
    """
    workers = (os.cpu_count() or 1) if n_jobs <= 0 else n_jobs
    workers = max(1, min(workers, periods.size // 1000 or 1))
    if workers == 1:
        return _power_chunk(t, y, dy, periods, durations, objective)
    chunks = np.array_split(periods, workers * 4)
    task = partial(_power_chunk, t, y, dy, durations=durations, objective=objective)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(task, chunks))
    return {k: np.concatenate([p[k] for p in parts]) for k in _PG_FIELDS}


def run_bls(lc: LightCurveData, cfg: BLSConfig) -> tuple[BLSResult, dict[str, FloatArray]]:
    """Run a BLS period search on a normalized (ideally detrended) light curve.

    Returns the best-peak summary and the full periodogram (``period``, ``power``).
    ``sde`` is the signal detection efficiency ``(peak - mean) / std`` of the periodogram,
    computed without any periodogram detrending.
    """
    finite = np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
    t, y, dy = lc.time[finite], lc.flux[finite], lc.flux_err[finite]
    if t.size < 100:
        raise ValueError(f"too few valid samples for BLS ({t.size})")
    baseline = float(t.max() - t.min())
    durations = np.asarray(sorted(cfg.durations_hours), dtype=np.float64) / 24.0
    max_period = min(cfg.max_period_days, baseline / (cfg.min_transits - 1))
    if max_period <= cfg.min_period_days:
        raise ValueError(f"baseline {baseline:.1f} d too short for {cfg.min_transits} transits")
    periods = period_grid(
        baseline, cfg.min_period_days, max_period, float(durations.min()), cfg.oversample
    )
    if periods.size > cfg.max_grid_size:
        raise ValueError(
            f"period grid has {periods.size} points (> max_grid_size={cfg.max_grid_size}); "
            "reduce the period range/oversample or raise the limit"
        )
    logger.info(
        "BLS: %d points, baseline %.1f d, %d periods in [%.3f, %.3f] d, %d durations",
        t.size,
        baseline,
        periods.size,
        periods[0],
        periods[-1],
        durations.size,
    )

    model = BoxLeastSquares(t, y, dy=dy)
    pg = _power(t, y, dy, periods, durations, cfg.objective, cfg.n_jobs)
    power = pg["power"]
    best = int(np.nanargmax(power))
    period = float(periods[best])
    duration = float(pg["duration"][best])
    t0 = float(pg["transit_time"][best])
    stats = model.compute_stats(period, duration, t0)
    per_transit = np.asarray(stats["per_transit_count"])
    std = float(np.nanstd(power))

    result = BLSResult(
        period_days=period,
        duration_hours=duration * 24.0,
        transit_time_bkjd=t0,
        depth=float(pg["depth"][best]),
        depth_err=float(pg["depth_err"][best]),
        depth_snr=float(pg["depth_snr"][best]),
        power=float(power[best]),
        log_likelihood=float(pg["log_likelihood"][best]),
        sde=float((power[best] - np.nanmean(power)) / std) if std > 0 else float("nan"),
        depth_odd=float(stats["depth_odd"][0]),
        depth_odd_err=float(stats["depth_odd"][1]),
        depth_even=float(stats["depth_even"][0]),
        depth_even_err=float(stats["depth_even"][1]),
        harmonic_delta_log_likelihood=float(stats["harmonic_delta_log_likelihood"]),
        n_transits_with_data=int((per_transit > 0).sum()),
        objective=cfg.objective,
        n_points=int(t.size),
        baseline_days=baseline,
        n_periods=int(periods.size),
        min_period_days=float(periods[0]),
        max_period_days=float(periods[-1]),
        durations_hours=[float(d * 24.0) for d in durations],
        top_peaks=_top_peaks(periods, power),
    )
    return result, {"period": periods, "power": power}


def downsample_periodogram(
    period: FloatArray, power: FloatArray, max_points: int = 2000
) -> dict[str, list[float]]:
    """Max-pool a periodogram to at most ``max_points`` for storage/plotting (peaks preserved)."""
    n = period.size
    if n <= max_points:
        return {"period": period.tolist(), "power": power.tolist()}
    edges = np.linspace(0, n, max_points + 1).astype(int)
    idx = [a + int(np.nanargmax(power[a:b])) for a, b in pairwise(edges) if b > a]
    return {"period": period[idx].tolist(), "power": power[idx].tolist()}


def compare_periods(
    bls_period: float,
    catalog_period: float,
    *,
    rel_tol: float = 1e-3,
    max_harmonic: int = 4,
) -> dict[str, Any]:
    """Relate a BLS period to a catalog period.

    ``relation`` is ``"match"`` if the ratio is 1 within ``rel_tol``, ``"harmonic"`` if it is
    ``n/m`` (n, m ≤ ``max_harmonic``, reduced fraction), otherwise ``"none"``. The simplest
    matching fraction wins. This is a bookkeeping check for one object, not a performance metric.
    """
    if bls_period <= 0 or catalog_period <= 0:
        raise ValueError("periods must be positive")
    ratio = bls_period / catalog_period
    candidates = sorted(
        {
            (n, m)
            for n in range(1, max_harmonic + 1)
            for m in range(1, max_harmonic + 1)
            if math.gcd(n, m) == 1
        },
        key=lambda nm: (nm[0] * nm[1], nm),
    )
    for n, m in candidates:
        target = n / m
        rel_err = abs(ratio - target) / target
        if rel_err <= rel_tol:
            return {
                "ratio": ratio,
                "relation": "match" if (n, m) == (1, 1) else "harmonic",
                "harmonic": f"{n}/{m}",
                "relative_error": rel_err,
            }
    return {
        "ratio": ratio,
        "relation": "none",
        "harmonic": None,
        "relative_error": abs(ratio - 1.0),
    }
