"""Per-target preprocessing: raw Kepler FITS → cleaned, normalized, detrended table.

Raw FITS files are never modified. The output is written to
``data/processed/lightcurves/kic_*/`` as ``lightcurve.parquet`` plus ``preprocessing.json``,
which records the configuration, input checksums and per-step sample counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoreliability import __version__
from exoreliability.config import PreprocessingConfig, ProjectPaths
from exoreliability.data.cache import (
    processed_target_dir,
    read_json,
    sha256_file,
    write_json_atomic,
)
from exoreliability.data.contracts import Ephemeris, LightCurveData
from exoreliability.data.mast import read_kepler_fits
from exoreliability.preprocessing.clean import clean, upper_outlier_mask
from exoreliability.preprocessing.detrend import robust_spline_trend, running_median_trend
from exoreliability.preprocessing.normalize import normalize_per_quarter
from exoreliability.preprocessing.phase_fold import in_transit_mask
from exoreliability.training.reproducibility import package_versions

PROCESSED_COLUMNS = (
    "time_bkjd",
    "quarter",
    "flux_norm",
    "flux_err_norm",
    "trend",
    "flux_detrended",
    "flux_err_detrended",
)


@dataclass
class ProcessedLightCurve:
    frame: pd.DataFrame
    record: dict[str, Any]

    def to_lightcurve(self, *, detrended: bool = True) -> LightCurveData:
        f = self.frame
        flux_col = "flux_detrended" if detrended else "flux_norm"
        err_col = "flux_err_detrended" if detrended else "flux_err_norm"
        return LightCurveData(
            time=f["time_bkjd"].to_numpy(np.float64),
            flux=f[flux_col].to_numpy(np.float64),
            flux_err=f[err_col].to_numpy(np.float64),
            quality=np.zeros(len(f), dtype=np.int64),
            quarter=f["quarter"].to_numpy(np.int64),
        )


def preprocess_files(
    fits_paths: list[Path],
    cfg: PreprocessingConfig,
    *,
    flux_column: str = "PDCSAP_FLUX",
    ephemerides: list[Ephemeris] | None = None,
) -> ProcessedLightCurve:
    """Run the configured preprocessing chain on one target's quarterly FITS files.

    Order: per-quarter clean (non-finite + quality bitmask) → per-quarter median
    normalization → concatenate → detrend (``running_median`` or ``robust_spline``,
    segment-aware) → optional clipping of *upward* outliers on the detrended flux.

    ``ephemerides`` are only used when ``cfg.detrend.mask_known_transits`` is true; then
    every sample within ±``mask_duration_factor``/2 catalog durations of any given
    ephemeris is excluded from the trend fit (column ``in_known_transit_mask``).
    """
    if not fits_paths:
        raise ValueError("no input files")
    parts: list[LightCurveData] = []
    per_quarter: list[dict[str, Any]] = []
    for path in sorted(fits_paths):
        raw = read_kepler_fits(path, flux_column=flux_column)
        cleaned, stats = clean(raw, cfg.quality_bitmask)
        if len(cleaned) == 0:
            per_quarter.append({**raw.meta, "clean": stats, "median_flux": None, "skipped": True})
            continue
        normed, medians = normalize_per_quarter(cleaned)
        parts.append(normed)
        per_quarter.append(
            {
                **raw.meta,
                "clean": stats,
                "median_flux": medians[raw.meta["quarter"]],
                "skipped": False,
            }
        )
    if not parts:
        raise ValueError("all quarters were empty after cleaning")
    lc = LightCurveData.concatenate(parts)

    exclude = None
    if cfg.detrend.mask_known_transits:
        if not ephemerides:
            raise ValueError("mask_known_transits requires ephemerides")
        exclude = np.zeros(len(lc), dtype=bool)
        for eph in ephemerides:
            if eph.duration_hours is not None:
                exclude |= in_transit_mask(
                    lc.time, eph, duration_factor=cfg.detrend.mask_duration_factor
                )

    d = cfg.detrend
    if not d.enabled:
        trend = np.ones_like(lc.flux)
    elif d.method == "robust_spline":
        trend = robust_spline_trend(
            lc.time,
            lc.flux,
            knot_spacing_days=d.knot_spacing_days,
            gap_days=d.gap_days,
            min_points=d.min_points,
            sigma_lower=d.sigma_lower,
            sigma_upper=d.sigma_upper,
            max_iter=d.max_iter,
            exclude=exclude,
        )
    else:
        trend = running_median_trend(
            lc.time,
            lc.flux,
            window_days=d.window_days,
            gap_days=d.gap_days,
            min_points=d.min_points,
            exclude=exclude,
        )
    flux_detrended = lc.flux / trend
    err_detrended = lc.flux_err / trend

    keep = np.ones(len(lc), dtype=bool)
    if cfg.clip_upper_sigma is not None:
        keep = ~upper_outlier_mask(flux_detrended, cfg.clip_upper_sigma)

    frame = pd.DataFrame(
        {
            "time_bkjd": lc.time[keep],
            "quarter": lc.quarter[keep].astype(np.int16),
            "flux_norm": lc.flux[keep],
            "flux_err_norm": lc.flux_err[keep],
            "trend": trend[keep],
            "flux_detrended": flux_detrended[keep],
            "flux_err_detrended": err_detrended[keep],
            "in_known_transit_mask": (exclude if exclude is not None else np.zeros(len(lc), bool))[
                keep
            ],
        }
    )
    record: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "exoreliability_version": __version__,
        "packages": package_versions(),
        "flux_column": flux_column,
        "time_system": "BKJD (BJD - 2454833)",
        "config": cfg.model_dump(),
        "detrend_masked_transits": bool(exclude is not None),
        "inputs": [
            {"file": Path(p).name, "sha256": sha256_file(Path(p))} for p in sorted(fits_paths)
        ],
        "quarters": per_quarter,
        "counts": {
            "after_clean": len(lc),
            "upper_outliers_clipped": int((~keep).sum()),
            "final": int(keep.sum()),
        },
    }
    return ProcessedLightCurve(frame, record)


def save_processed(processed: ProcessedLightCurve, paths: ProjectPaths, kepid: int) -> Path:
    out = processed_target_dir(paths, kepid)
    out.mkdir(parents=True, exist_ok=True)
    processed.frame.to_parquet(out / "lightcurve.parquet", index=False)
    write_json_atomic(out / "preprocessing.json", {"kepid": kepid, **processed.record})
    return out


def load_processed(paths: ProjectPaths, kepid: int) -> ProcessedLightCurve | None:
    out = processed_target_dir(paths, kepid)
    pq, meta = out / "lightcurve.parquet", out / "preprocessing.json"
    if not (pq.exists() and meta.exists()):
        return None
    return ProcessedLightCurve(pd.read_parquet(pq), read_json(meta))
