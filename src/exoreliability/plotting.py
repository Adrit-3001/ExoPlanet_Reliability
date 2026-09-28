"""Static diagnostic figures (matplotlib, non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoreliability.data.contracts import Ephemeris
from exoreliability.preprocessing.phase_fold import phase_fold
from exoreliability.preprocessing.resample import bin_curve


def plot_target_diagnostic(
    frame: pd.DataFrame,
    *,
    kepid: int,
    kois: list[tuple[str, str, Ephemeris]],
    out_path: Path,
    phase_window_hours: float = 24.0,
    n_bins: int = 120,
) -> Path:
    """Normalized light curve, detrended light curve, and one phase fold per KOI.

    ``kois`` holds ``(kepoi_name, catalog disposition, catalog ephemeris)``. Folding uses the
    catalog ephemeris, so these panels visualise the catalog signal, not a detection.
    """
    n_fold = max(1, len(kois))
    fig = plt.figure(figsize=(12, 6 + 3 * ((n_fold + 1) // 2)), constrained_layout=True)
    grid = fig.add_gridspec(2 + (n_fold + 1) // 2, 2)
    t = frame["time_bkjd"].to_numpy()

    ax = fig.add_subplot(grid[0, :])
    ax.plot(t, frame["flux_norm"], ",", color="0.3", rasterized=True)
    # Break the trend line at data gaps (> 0.5 d) instead of drawing across them.
    gaps = np.flatnonzero(np.diff(t) > 0.5) + 1
    ax.plot(
        np.insert(t, gaps, np.nan),
        np.insert(frame["trend"].to_numpy(), gaps, np.nan),
        "-",
        color="tab:orange",
        lw=0.8,
        label="running-median trend",
    )
    ax.set(
        title=f"KIC {kepid} — PDCSAP flux, per-quarter median normalized",
        xlabel="Time [BKJD]",
        ylabel="Normalized flux",
    )
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(grid[1, :])
    ax.plot(t, frame["flux_detrended"], ",", color="0.3", rasterized=True)
    ax.set(title="Detrended flux", xlabel="Time [BKJD]", ylabel="Relative flux")

    for i, (name, disposition, eph) in enumerate(kois):
        ax = fig.add_subplot(grid[2 + i // 2, i % 2])
        hours = phase_fold(t, eph) * 24.0
        window = min(phase_window_hours, 12.0 * eph.period_days)
        sel = np.abs(hours) <= window
        y = frame["flux_detrended"].to_numpy()
        ax.plot(hours[sel], y[sel], ".", ms=1.5, color="0.6", alpha=0.5, rasterized=True)
        centres, med, _ = bin_curve(hours[sel], y[sel], n_bins, (-window, window))
        ax.plot(centres, med, "-", color="tab:blue", lw=1.2, label="binned median")
        ax.set(
            title=f"{name} ({disposition}) folded on catalog P = {eph.period_days:.5f} d",
            xlabel="Hours from catalog mid-transit",
            ylabel="Relative flux",
        )
        ax.legend(loc="lower right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_bls_periodograms(
    periodograms: list[tuple[str, np.ndarray, np.ndarray, float]],
    *,
    title: str,
    catalog_periods: list[tuple[str, float]],
    out_path: Path,
) -> Path:
    """Stacked BLS periodograms (``label, period, power, best_period``) with catalog periods marked."""
    fig, axes = plt.subplots(
        len(periodograms),
        1,
        figsize=(11, 2.2 * len(periodograms)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (label, period, power, best) in zip(axes[:, 0], periodograms, strict=True):
        ax.plot(period, power, "-", color="0.25", lw=0.6)
        for name, p in catalog_periods:
            ax.axvline(p, color="tab:red", ls="--", lw=0.8, alpha=0.7, label=f"catalog {name}")
        ax.axvline(best, color="tab:blue", ls=":", lw=1.0, label=f"BLS best {best:.4f} d")
        ax.set_xscale("log")
        ax.set_ylabel("BLS power")
        ax.set_title(label, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("Period [days]")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
