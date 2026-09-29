"""Request/response contracts for target, light-curve and BLS endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    catalog_snapshot_available: bool
    processed_targets: int


class KOI(BaseModel):
    """One Kepler Object of Interest from the DR25 KOI table (units as in the archive)."""

    kepoi_name: str
    kepler_name: str | None = None
    koi_disposition: str | None = Field(None, description="Exoplanet Archive disposition")
    koi_pdisposition: str | None = Field(None, description="DR25 disposition using Kepler data")
    label_name: str | None = Field(None, description="Derived label (policy dr25_clean_v2)")
    exclusion_reason: str | None = None
    koi_score: float | None = None
    koi_period: float | None = Field(None, description="days")
    koi_time0bk: float | None = Field(None, description="BKJD")
    koi_duration: float | None = Field(None, description="hours")
    koi_depth: float | None = Field(None, description="ppm")
    koi_model_snr: float | None = None
    koi_prad: float | None = Field(None, description="Earth radii")
    koi_kepmag: float | None = None


class TargetSummary(BaseModel):
    target_id: str
    kepid: int
    n_kois: int
    dispositions: list[str]
    has_lightcurve: bool
    has_bls: bool


class TargetList(BaseModel):
    targets: list[TargetSummary]


class ProductSummary(BaseModel):
    n_products: int
    quarters: list[int]
    author: str
    exptime_seconds: float
    retrieved_at_utc: str | None


class PreprocessingSummary(BaseModel):
    created_at_utc: str | None
    flux_column: str
    n_points: int
    data_release: list[str]
    config: dict[str, object]


class CatalogSource(BaseModel):
    table: str
    retrieved_at_utc: str
    query: str


class BLSStats(BaseModel):
    period_days: float
    duration_hours: float
    transit_time_bkjd: float
    depth: float = Field(description="fractional depth")
    depth_err: float
    depth_snr: float
    power: float
    log_likelihood: float
    sde: float
    depth_odd: float
    depth_even: float
    harmonic_delta_log_likelihood: float
    n_transits_with_data: int
    n_points: int
    baseline_days: float
    n_periods: int
    min_period_days: float
    max_period_days: float


class PeriodComparison(BaseModel):
    kepoi_name: str
    catalog_period: float
    ratio: float
    relation: Literal["match", "harmonic", "none"]
    harmonic: str | None
    relative_error: float


class BLSRun(BaseModel):
    perturbation: str
    severity: float
    sigma_added: float
    stats: BLSStats
    catalog_comparison: list[PeriodComparison]


class Periodogram(BaseModel):
    period: list[float]
    power: list[float]
    note: str = "max-pooled for display"


class BLSResponse(BaseModel):
    kepid: int
    run_id: str
    clean: BLSRun | None
    noise_demo: list[BLSRun]
    periodogram: Periodogram | None
    note: str


class TargetDetail(TargetSummary):
    kois: list[KOI]
    catalog_source: CatalogSource | None
    products: ProductSummary | None
    preprocessing: PreprocessingSummary | None
    bls_clean: BLSRun | None
    bls_run_id: str | None


class LightCurveResponse(BaseModel):
    kepid: int
    time_system: str = "BKJD (BJD - 2454833)"
    n_points: int
    binned: bool
    bin_width_days: float | None
    time: list[float]
    flux_norm: list[float]
    trend: list[float]
    flux_detrended: list[float]
    quarter: list[int]


class PhaseFoldResponse(BaseModel):
    kepid: int
    kepoi_name: str | None
    ephemeris_source: Literal["catalog", "bls"]
    period_days: float
    epoch_bkjd: float
    duration_hours: float | None
    window_hours: float
    phase_hours: list[float]
    flux: list[float]
    binned_phase_hours: list[float]
    binned_flux: list[float | None]
