"""Additive white Gaussian observational noise.

Severity semantics
------------------
``severity = s`` adds independent zero-mean Gaussian noise with standard deviation

    sigma_added = s × sigma_ref

to every flux sample, where ``sigma_ref`` is the light curve's own point-to-point
scatter, estimated robustly as ``1.4826 × MAD(diff(flux)) / sqrt(2)`` (insensitive to slow
trends and to the few in-transit points). A caller may instead supply
``metadata["sigma_ref"]`` to use a fixed reference level (e.g. to perturb several
light curves on a common absolute scale).

If the original noise is white with std sigma_ref, the total scatter becomes
``sigma_ref × sqrt(1 + s²)``: s = 1 roughly corresponds to a √2 worse noise level,
i.e. a star ≈ 0.75 mag fainter in the photon-noise-limited regime.

``severity = 0`` returns an unchanged copy and does not draw from ``rng``.

When ``update_flux_err`` is true (default), reported uncertainties are inflated in
quadrature, ``sqrt(flux_err² + sigma_added²)``, so downstream error-weighted methods see a
consistent noise model. Sample count and time stamps are always preserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from exoreliability.data.contracts import FloatArray, LightCurveData
from exoreliability.perturbations.base import PerturbationResult
from exoreliability.preprocessing.clean import MAD_TO_SIGMA


def point_to_point_sigma(flux: FloatArray) -> float:
    """Robust white-noise level from first differences: ``1.4826·MAD(Δflux)/√2``."""
    f = np.asarray(flux, dtype=np.float64)
    f = f[np.isfinite(f)]
    if f.size < 3:
        raise ValueError("need at least 3 finite samples to estimate noise")
    d = np.diff(f)
    return float(MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


@dataclass(frozen=True)
class GaussianNoise:
    name: str = "gaussian_noise"
    update_flux_err: bool = True

    def apply(
        self,
        light_curve: LightCurveData,
        severity: float,
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> PerturbationResult:
        if not np.isfinite(severity) or severity < 0:
            raise ValueError(f"severity must be a finite value >= 0, got {severity}")
        meta = dict(metadata or {})
        sigma_ref = (
            float(meta["sigma_ref"])
            if "sigma_ref" in meta
            else point_to_point_sigma(light_curve.flux)
        )
        sigma_source = "metadata" if "sigma_ref" in meta else "point_to_point_mad"
        if not np.isfinite(sigma_ref) or sigma_ref < 0:
            raise ValueError(f"invalid reference noise level {sigma_ref}")
        sigma_added = severity * sigma_ref
        params = {
            "perturbation": self.name,
            "severity": float(severity),
            "sigma_ref": sigma_ref,
            "sigma_ref_source": sigma_source,
            "sigma_added": sigma_added,
            "update_flux_err": self.update_flux_err,
        }
        if severity == 0:
            return PerturbationResult(_copy(light_curve), params)

        noise = rng.normal(0.0, sigma_added, size=len(light_curve))
        flux_err = (
            np.sqrt(light_curve.flux_err**2 + sigma_added**2)
            if self.update_flux_err
            else light_curve.flux_err.copy()
        )
        perturbed = LightCurveData(
            time=light_curve.time.copy(),
            flux=light_curve.flux + noise,
            flux_err=flux_err,
            quality=light_curve.quality.copy(),
            quarter=light_curve.quarter.copy(),
            meta={**light_curve.meta, "perturbation": params},
        )
        return PerturbationResult(perturbed, params)


def _copy(lc: LightCurveData) -> LightCurveData:
    return LightCurveData(
        lc.time.copy(),
        lc.flux.copy(),
        lc.flux_err.copy(),
        lc.quality.copy(),
        lc.quarter.copy(),
        dict(lc.meta),
    )
