"""Common interface for light-curve perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from exoreliability.data.contracts import LightCurveData


@dataclass(frozen=True)
class PerturbationResult:
    """Perturbed light curve plus every parameter needed to reproduce/interpret it."""

    light_curve: LightCurveData
    params: dict[str, Any] = field(default_factory=dict)


class Perturbation(Protocol):
    """A controlled, seeded modification of a light curve.

    Implementations must: take all randomness from ``rng``; treat ``severity == 0`` as the
    identity where meaningful; preserve the number of samples unless documented otherwise;
    and return every applied parameter in ``PerturbationResult.params``.
    """

    name: str

    def apply(
        self,
        light_curve: LightCurveData,
        severity: float,
        rng: np.random.Generator,
        metadata: Mapping[str, Any] | None = None,
    ) -> PerturbationResult: ...
