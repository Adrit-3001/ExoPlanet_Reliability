"""Shared fixtures. All data here is SYNTHETIC; no test performs network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from exoreliability.config import ProjectPaths
from exoreliability.data.contracts import LightCurveData

KEPLER_LC_CADENCE_DAYS = 0.0204335  # 29.4 min


def make_transit_lc(
    *,
    n_days: float = 90.0,
    period: float = 3.7,
    epoch: float = 1.3,
    duration_days: float = 0.12,
    depth: float = 1e-3,
    noise: float = 2e-4,
    seed: int = 0,
    t_start: float = 130.0,
) -> LightCurveData:
    """Synthetic flat light curve with box-shaped transits (BKJD times)."""
    rng = np.random.default_rng(seed)
    t = t_start + np.arange(0.0, n_days, KEPLER_LC_CADENCE_DAYS)
    flux = 1.0 + rng.normal(0.0, noise, t.size)
    phase = np.mod(t - (t_start + epoch) + 0.5 * period, period) - 0.5 * period
    flux[np.abs(phase) < duration_days / 2] -= depth
    return LightCurveData(
        time=t,
        flux=flux,
        flux_err=np.full(t.size, noise),
        quality=np.zeros(t.size, dtype=np.int64),
        quarter=np.ones(t.size, dtype=np.int64),
    )


@pytest.fixture
def tmp_paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(root=tmp_path)


def write_fake_kepler_fits(
    path: Path,
    *,
    kepid: int,
    quarter: int,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> Path:
    """Minimal file with the header keys and columns ``read_kepler_fits`` uses (SYNTHETIC)."""
    n = time.size
    primary = fits.PrimaryHDU()
    primary.header.update(
        {
            "TELESCOP": "Kepler",
            "KEPLERID": kepid,
            "QUARTER": quarter,
            "DATA_REL": 25,
            "OBSMODE": "long cadence",
            "OBJECT": f"KIC {kepid}",
        }
    )
    cols = [
        fits.Column("TIME", "D", array=time),
        fits.Column("PDCSAP_FLUX", "E", array=flux),
        fits.Column(
            "PDCSAP_FLUX_ERR", "E", array=np.full(n, 30.0) if flux_err is None else flux_err
        ),
        fits.Column(
            "SAP_QUALITY", "J", array=np.zeros(n, dtype=np.int32) if quality is None else quality
        ),
    ]
    table = fits.BinTableHDU.from_columns(cols)
    table.header.update({"BJDREFI": 2454833, "BJDREFF": 0.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return path
