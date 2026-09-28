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


def get_paths(root: Path | None = None) -> ProjectPaths:
    return ProjectPaths(root=(root or _find_project_root()))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


class CatalogConfig(_Strict):
    table: Literal["q1_q17_dr25_koi"] = "q1_q17_dr25_koi"
    label_rule: Literal["dr25_conservative_v1"] = "dr25_conservative_v1"


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


class DataConfig(_Strict):
    name: str
    catalog: CatalogConfig = CatalogConfig()
    selection: SelectionConfig = SelectionConfig()
    lightcurves: LightCurveConfig = LightCurveConfig()


# ---------------------------------------------------------------------------
# Preprocessing / BLS experiment configuration
# ---------------------------------------------------------------------------


class DetrendConfig(_Strict):
    enabled: bool = True
    method: Literal["running_median"] = "running_median"
    window_days: float = Field(default=1.5, gt=0)
    gap_days: float = Field(default=0.5, gt=0)
    min_points: int = Field(default=10, ge=1)
    mask_known_transits: bool = False


class PreprocessingConfig(_Strict):
    quality_bitmask: int = Field(default=KEPLER_DEFAULT_QUALITY_BITMASK, ge=0)
    normalize: Literal["per_quarter_median"] = "per_quarter_median"
    detrend: DetrendConfig = DetrendConfig()
    clip_upper_sigma: float | None = Field(default=5.0, gt=0)


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
