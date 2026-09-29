"""Fixed-length phase-folded views of one KOI signal.

* **Global view**: ``global_bins`` non-overlapping bins covering the full orbital phase
  [-0.5, 0.5) (mid-transit at 0). Captures secondary eclipses and out-of-transit shape.
* **Local view**: ``local_bins`` bin centres spanning ±``local_half_width_durations``
  transit durations, each bin ``local_bin_width_durations`` durations wide (bins overlap
  when the width exceeds the spacing). Resolves the transit shape independently of the
  period, which the global view cannot do for long periods.

Each bin holds the **median** of ``flux - 1`` of the samples falling in it (NaN if none)
and the sample **count**. Empty bins are left as NaN here; filling is a separate,
documented step in ``training.dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from exoreliability.data.contracts import Ephemeris, FloatArray
from exoreliability.preprocessing.phase_fold import in_transit_mask, phase_fold


@dataclass(frozen=True)
class View:
    centres: FloatArray  # global: phase fraction; local: transit durations
    flux: NDArray[np.float32]  # median(flux - 1) per bin, NaN where empty
    count: NDArray[np.int32]

    @property
    def coverage(self) -> float:
        return float(np.mean(self.count > 0))


def windowed_median(
    x: FloatArray, y: FloatArray, centres: FloatArray, width: float
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Median and count of ``y`` over half-open windows ``[c - width/2, c + width/2)``."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    lo = np.searchsorted(xs, centres - 0.5 * width, side="left")
    hi = np.searchsorted(xs, centres + 0.5 * width, side="left")
    counts = (hi - lo).astype(np.int32)
    med = np.full(centres.size, np.nan, dtype=np.float32)
    for i in np.flatnonzero(counts):
        med[i] = np.median(ys[lo[i] : hi[i]])
    return med, counts


def global_view(time: FloatArray, flux: FloatArray, eph: Ephemeris, n_bins: int) -> View:
    phase = phase_fold(time, eph) / eph.period_days  # [-0.5, 0.5)
    width = 1.0 / n_bins
    centres = np.asarray(-0.5 + width * (np.arange(n_bins) + 0.5), dtype=np.float64)
    med, cnt = windowed_median(phase, flux - 1.0, centres, width)
    return View(centres, med, cnt)


def local_view(
    time: FloatArray,
    flux: FloatArray,
    eph: Ephemeris,
    n_bins: int,
    half_width_durations: float,
    bin_width_durations: float,
) -> View:
    if eph.duration_hours is None or eph.duration_hours <= 0:
        raise ValueError("local view requires a positive duration")
    x = phase_fold(time, eph) * 24.0 / eph.duration_hours  # in transit durations
    centres = np.linspace(-half_width_durations, half_width_durations, n_bins)
    med, cnt = windowed_median(x, flux - 1.0, centres, bin_width_durations)
    return View(centres, med, cnt)


def transit_coverage(view: View, eph: Ephemeris) -> float:
    """Fraction of global-view bins inside the transit (|phase| < D/2P) that are observed.

    Uses at least the single bin nearest mid-transit when the transit is narrower than a bin.
    """
    assert eph.duration_hours is not None
    half = 0.5 * eph.duration_hours / 24.0 / eph.period_days
    inside = np.abs(view.centres) < half
    if not inside.any():
        inside = np.abs(view.centres) == np.abs(view.centres).min()
    return float(np.mean(view.count[inside] > 0))


def other_koi_contamination(
    time: FloatArray,
    eph: Ephemeris,
    own_depth_ppm: float | None,
    others: list[tuple[Ephemeris, float | None]],
) -> dict[str, float]:
    """How strongly *other* KOIs' transits survive folding on this KOI's ephemeris.

    The phase is divided into bins of width D/2 (this KOI's duration), so every bin holds
    many samples. For each bin and each other KOI B, ``frac_B`` is the fraction of the bin's
    samples inside B's transit (±D_B/2); ``frac_B × depth_B`` (catalog depth, box
    approximation) is the dip B contributes to the bin **mean**. Incoherent signals spread
    over all phases and contribute ~duty-cycle × depth; coherent ones (same or
    commensurate period) pile up in a few bins.

    Returns the maximum fraction over bins, the maximum expected contaminating dip (ppm)
    over bins and within this KOI's own transit, and that maximum relative to this KOI's
    own catalog depth.
    """
    empty = {
        "contam_max_bin_fraction": 0.0,
        "contam_max_expected_ppm": 0.0,
        "contam_in_transit_expected_ppm": 0.0,
        "contam_max_relative_to_depth": 0.0,
    }
    others = [(o, d) for o, d in others if o.duration_hours]
    if not others or not eph.duration_hours:
        return empty
    n_bins = int(np.clip(np.ceil(2 * eph.period_days * 24.0 / eph.duration_hours), 8, 4096))
    phase = phase_fold(time, eph) / eph.period_days
    idx = np.clip(((phase + 0.5) * n_bins).astype(int), 0, n_bins - 1)
    total = np.bincount(idx, minlength=n_bins).astype(float)
    populated = total > 0
    union = np.zeros(time.size, dtype=bool)
    amp = np.zeros(n_bins)
    for o, depth in others:
        hit = in_transit_mask(time, o, duration_factor=1.0)
        union |= hit
        frac = np.divide(
            np.bincount(idx, weights=hit.astype(float), minlength=n_bins),
            total,
            out=np.zeros(n_bins),
            where=populated,
        )
        if depth is not None and np.isfinite(depth):
            amp += frac * depth
    frac_union = np.divide(
        np.bincount(idx, weights=union.astype(float), minlength=n_bins),
        total,
        out=np.zeros(n_bins),
        where=populated,
    )
    centres = -0.5 + (np.arange(n_bins) + 0.5) / n_bins
    own_bins = np.abs(centres) < 0.5 * eph.duration_hours / 24.0 / eph.period_days + 0.5 / n_bins
    max_amp = float(amp.max())
    return {
        "contam_max_bin_fraction": float(frac_union.max()),
        "contam_max_expected_ppm": max_amp,
        "contam_in_transit_expected_ppm": float(amp[own_bins].max()) if own_bins.any() else 0.0,
        "contam_max_relative_to_depth": max_amp / own_depth_ppm
        if own_depth_ppm and np.isfinite(own_depth_ppm) and own_depth_ppm > 0
        else float("nan"),
    }
