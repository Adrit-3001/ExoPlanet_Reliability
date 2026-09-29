"""Project paths and validated YAML configuration models.

Every experiment parameter that affects results lives in a YAML file under ``configs/``
and is validated here before any network access or computation happens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Lightkurve's "default" Kepler quality bitmask (KeplerQualityFlags.DEFAULT_BITMASK).
# Stored as an explicit integer so the preprocessing record does not depend on a library default.
KEPLER_DEFAULT_QUALITY_BITMASK = 1130799


def _find_project_root() -> Path:
    env = os.environ.get("EXORELIABILITY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout for data and artifacts, rooted at ``root``."""

    root: Path

    @property
    def raw_catalogs(self) -> Path:
        return self.root / "data" / "raw" / "catalogs"

    @property
    def raw_lightcurves(self) -> Path:
        return self.root / "data" / "raw" / "lightcurves"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_lightcurves(self) -> Path:
        return self.root / "data" / "processed" / "lightcurves"

    @property
    def experiments(self) -> Path:
        return self.root / "artifacts" / "experiments"

    @property
    def figures(self) -> Path:
        return self.root / "artifacts" / "figures"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def splits(self) -> Path:
        return self.root / "data" / "splits"

    @property
    def datasets(self) -> Path:
        return self.root / "data" / "processed" / "datasets"

    @property
    def reports(self) -> Path:
        return self.root / "artifacts" / "reports"


def get_paths(root: Path | None = None) -> ProjectPaths:
    return ProjectPaths(root=(root or _find_project_root()))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


class CatalogConfig(_Strict):
    table: Literal["q1_q17_dr25_koi"] = "q1_q17_dr25_koi"
    label_rule: Literal["dr25_clean_v2"] = "dr25_clean_v2"


class SelectionConfig(_Strict):
    """Deterministic selection of a small KOI subset from the cached catalog snapshot."""

    seed: int = 42
    limit: int = Field(default=20, ge=1)
    stratify_by_label: bool = True
    include_excluded: bool = False


class LightCurveConfig(_Strict):
    """Product policy for MAST light-curve retrieval."""

    mission: Literal["Kepler"] = "Kepler"
    author: Literal["Kepler"] = "Kepler"
    exptime_seconds: Literal[1800] = 1800  # Kepler long cadence
    flux_column: Literal["PDCSAP_FLUX", "SAP_FLUX"] = "PDCSAP_FLUX"
    quarters: list[int] | None = None  # None = all available long-cadence quarters
    max_targets: int = Field(default=3, ge=1)
    max_products_per_target: int = Field(default=18, ge=1)
    max_total_download_mb: float = Field(default=250.0, gt=0)

    @field_validator("quarters")
    @classmethod
    def _valid_quarters(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(q < 0 or q > 17 for q in v):
            raise ValueError("Kepler quarters must be within 0..17")
        return v


# ---------------------------------------------------------------------------
# Preprocessing / BLS experiment configuration
# ---------------------------------------------------------------------------


class DetrendConfig(_Strict):
    """Trend removal. ``running_median`` uses ``window_days``; ``robust_spline`` uses
    ``knot_spacing_days`` and iterative sigma clipping. ``mask_known_transits`` excludes
    ±``mask_duration_factor``/2 catalog durations around every known KOI transit on the
    star from the trend fit (the trend is still evaluated there)."""

    enabled: bool = True
    method: Literal["running_median", "robust_spline"] = "running_median"
    window_days: float = Field(default=1.5, gt=0)
    knot_spacing_days: float = Field(default=0.3, gt=0)
    sigma_lower: float = Field(default=3.0, gt=0)
    sigma_upper: float = Field(default=3.0, gt=0)
    max_iter: int = Field(default=3, ge=1)
    gap_days: float = Field(default=0.5, gt=0)
    min_points: int = Field(default=10, ge=1)
    mask_known_transits: bool = False
    mask_duration_factor: float = Field(default=2.0, gt=0)


class PreprocessingConfig(_Strict):
    quality_bitmask: int = Field(default=KEPLER_DEFAULT_QUALITY_BITMASK, ge=0)
    normalize: Literal["per_quarter_median"] = "per_quarter_median"
    detrend: DetrendConfig = DetrendConfig()
    clip_upper_sigma: float | None = Field(default=5.0, gt=0)


class StarSelectionConfig(_Strict):
    """Which stars enter a dataset build. Stars are the sampling unit; every KOI on a
    selected star becomes an example (candidates included, flagged as excluded)."""

    seed: int = 42
    all_eligible: bool = False  # full scale: every star with >= 1 train-eligible KOI
    n_stars: int = Field(default=0, ge=0)  # sampled stars, in addition to include_kepids
    include_kepids: list[int] = Field(default_factory=list)


class RepresentationConfig(_Strict):
    """Fixed-length phase-folded views (see docs/data_contract.md §7)."""

    global_bins: int = Field(default=2048, ge=16)
    local_bins: int = Field(default=201, ge=11)
    local_half_width_durations: float = Field(default=2.0, gt=0)
    local_bin_width_durations: float = Field(default=0.16, gt=0)
    min_valid_points: int = Field(default=1000, ge=1)
    min_global_coverage: float = Field(default=0.5, ge=0, le=1)
    min_transit_coverage: float = Field(default=0.5, ge=0, le=1)
    # Longest masked window (mask_duration_factor x duration) for which masked detrending
    # was validated on synthetic data; longer windows get example_status detrend_mask_too_long.
    max_mask_window_days: float = Field(default=2.0, gt=0)


class DatasetBuildConfig(_Strict):
    scale: Literal["smoke", "small", "full"]
    splits_dir: str = "data/splits"
    allow_download: bool = True
    stars: StarSelectionConfig = StarSelectionConfig()
    preprocessing: PreprocessingConfig
    representation: RepresentationConfig = RepresentationConfig()


class DataConfig(_Strict):
    name: str
    catalog: CatalogConfig = CatalogConfig()
    selection: SelectionConfig = SelectionConfig()
    lightcurves: LightCurveConfig = LightCurveConfig()
    dataset: DatasetBuildConfig | None = None


class SplitConfig(_Strict):
    """Star-level (KIC-grouped) train/val/test assignment over the whole catalog snapshot."""

    name: str = "kic_grouped_v1"
    seed: int = 42
    label_policy: Literal["dr25_clean_v2"] = "dr25_clean_v2"
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    val_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1 (got {total})")
        return self

    @property
    def fractions(self) -> dict[str, float]:
        return {"train": self.train_fraction, "val": self.val_fraction, "test": self.test_fraction}


class BLSConfig(_Strict):
    min_period_days: float = Field(default=0.5, gt=0)
    max_period_days: float = Field(default=50.0, gt=0)
    min_transits: int = Field(default=3, ge=2)
    durations_hours: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    oversample: float = Field(default=3.0, gt=0)
    objective: Literal["likelihood", "snr"] = "likelihood"
    max_grid_size: int = Field(default=2_000_000, ge=100)
    # Worker processes for the periodogram (<= 0 means all CPUs). Does not change results.
    n_jobs: int = 0

    @model_validator(mode="after")
    def _check(self) -> BLSConfig:
        if self.max_period_days <= self.min_period_days:
            raise ValueError("max_period_days must exceed min_period_days")
        if not self.durations_hours:
            raise ValueError("durations_hours must be non-empty")
        if max(self.durations_hours) / 24.0 >= self.min_period_days:
            raise ValueError("every BLS duration must be shorter than min_period_days")
        return self


class NoiseDemoConfig(_Strict):
    enabled: bool = True
    severities: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    seed: int = 42

    @field_validator("severities")
    @classmethod
    def _non_negative(cls, v: list[float]) -> list[float]:
        if any(s < 0 for s in v):
            raise ValueError("severities must be >= 0")
        return v


class BLSExperimentConfig(_Strict):
    name: str
    seed: int = 42
    data_config: str
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    bls: BLSConfig = BLSConfig()
    noise_demo: NoiseDemoConfig = NoiseDemoConfig()


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load and validate a YAML config; raises ``pydantic.ValidationError`` on bad input."""
    return model.model_validate(load_yaml(path))


def resolve_path(path: str | Path, paths: ProjectPaths) -> Path:
    p = Path(path)
    return p if p.is_absolute() else paths.root / p
