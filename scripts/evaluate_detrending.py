"""Benchmark transit preservation of detrending methods (synthetic) and compare on real stars.

Synthetic part: every case in exoreliability.evaluation.detrending.CASES × seeds × methods.
Real part: for locally downloaded stars, fold each KOI on its catalog ephemeris after
(a) the Milestone 1 method (blind running median, 1.5 d) and (b) the dataset method
(masked robust spline), and compare the measured depths. Real stars have no ground truth;
the catalog depth is shown only for context.

Outputs: artifacts/reports/detrending/{synthetic_results.csv, synthetic_summary.csv,
real_comparison.csv, summary.json} and artifacts/figures/detrending_synthetic_cases.png.

Example:
    python scripts/evaluate_detrending.py --seeds 10 --real-limit 20
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoreliability.config import (
    DataConfig,
    DetrendConfig,
    PreprocessingConfig,
    get_paths,
    load_config,
    resolve_path,
)
from exoreliability.data.cache import write_json_atomic
from exoreliability.data.catalog import ephemeris_from_row
from exoreliability.data.contracts import Ephemeris
from exoreliability.evaluation.detrending import (
    ACCEPTANCE_CASES,
    CASES,
    DepthResult,
    evaluate,
    make_case,
    method_registry,
)
from exoreliability.logging import setup_logging
from exoreliability.preprocessing.phase_fold import in_transit_mask, phase_fold
from exoreliability.preprocessing.pipeline import preprocess_files
from exoreliability.preprocessing.resample import bin_curve
from exoreliability.targets import TargetRepository

logger = logging.getLogger("evaluate_detrending")
MEAN_TOL, RMS_TOL = 0.02, 0.05


def folded_depth_ppm(time: np.ndarray, flat: np.ndarray, eph: Ephemeris) -> float:
    """Mean depth in the central 50 % of the transit relative to the median of the
    out-of-transit annulus between 1 and 3 durations from mid-transit."""
    assert eph.duration_hours is not None
    x = np.abs(phase_fold(time, eph)) * 24.0 / eph.duration_hours
    inner = x < 0.25
    annulus = (x > 1.0) & (x < 3.0)
    if inner.sum() < 3 or annulus.sum() < 10:
        return float("nan")
    return float((np.median(flat[annulus]) - np.mean(flat[inner])) * 1e6)


def synthetic(seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[DepthResult] = []
    for case in CASES:
        for seed in range(seeds):
            lc = make_case(case, seed)
            for name, (fn, masked) in method_registry().items():
                r, _ = evaluate(lc, name, fn, masked)
                rows.append(DepthResult(**{**r.to_dict(), "seed": seed}))  # type: ignore[arg-type]
    df = pd.DataFrame([r.to_dict() for r in rows])
    summary = (
        df.groupby(["case", "method"])
        .agg(
            mean_relative_depth_error=("relative_depth_error", "mean"),
            rms_relative_depth_error=(
                "relative_depth_error",
                lambda s: float(np.sqrt(np.mean(s**2))),
            ),
            max_abs_relative_depth_error=("relative_depth_error", lambda s: float(s.abs().max())),
            trend_error_rms_ppm=("trend_error_rms_ppm", "mean"),
        )
        .reset_index()
    )
    summary["acceptance_case"] = summary.case.isin(ACCEPTANCE_CASES)
    summary["passes"] = (summary.mean_relative_depth_error.abs() <= MEAN_TOL) & (
        summary.rms_relative_depth_error <= RMS_TOL
    )
    return df, summary


def plot_cases(out_path) -> None:
    names = [
        "low_variability",
        "moderate_variability",
        "strong_variability",
        "strong_gaps_near_transits",
        "strong_long_transit",
    ]
    methods = method_registry()
    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(14, 3.0 * len(names)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    for row, name in enumerate(names):
        case = next(c for c in CASES if c.name == name)
        lc = make_case(case, 0)
        _, new = evaluate(lc, "robust_spline_masked", *methods["robust_spline_masked"])
        _, old = evaluate(lc, "running_median_blind", *methods["running_median_blind"])
        t0 = lc.ephemeris.epoch_bkjd + 3 * lc.ephemeris.period_days
        win = (lc.time > t0 - 1.6 * lc.ephemeris.period_days) & (
            lc.time < t0 + 1.6 * lc.ephemeris.period_days
        )
        intr = in_transit_mask(lc.time, lc.ephemeris, 1.0)
        ax = axes[row, 0]
        ax.plot(lc.time[win], lc.flux[win], ".", ms=1.5, color="0.6", label="raw signal")
        ax.plot(lc.time[win], lc.true_trend[win], "-", color="k", lw=1.2, label="true trend")
        ax.plot(
            lc.time[win], new[win], "-", color="tab:blue", lw=1, label="estimated: masked spline"
        )
        ax.plot(
            lc.time[win],
            old[win],
            "--",
            color="tab:orange",
            lw=1,
            label="estimated: M1 running median",
        )
        for tt in lc.time[win & intr]:
            ax.axvspan(tt - 0.01, tt + 0.01, color="tab:red", alpha=0.05, lw=0)
        ax.set_title(
            f"{name} (SYNTHETIC; variability {case.variability_ppm:.0f} ppm, P_rot {case.rotation_days} d, duration {case.duration_hours} h)",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("flux")
        if row == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax = axes[row, 1]
        hours = phase_fold(lc.time, lc.ephemeris) * 24
        sel = np.abs(hours) < 3 * case.duration_hours
        ax.plot(
            hours[sel],
            (lc.flux / lc.true_trend)[sel],
            ".",
            ms=1.5,
            color="0.7",
            label="flux / true trend",
        )
        span = (-3 * case.duration_hours, 3 * case.duration_hours)
        for est, color, lab in (
            (new, "tab:blue", "masked spline"),
            (old, "tab:orange", "M1 running median"),
        ):
            centres, med, _ = bin_curve(hours[sel], (lc.flux / est)[sel], 60, span)
            ax.plot(
                centres, med, "-", color=color, lw=1.4, label=f"flattened: {lab} (binned median)"
            )
        depth = case.depth_ppm * 1e-6
        ax.set_ylim(1 - 2.2 * depth, 1 + 1.2 * depth)
        ax.axvspan(
            -case.duration_hours / 2,
            case.duration_hours / 2,
            color="tab:red",
            alpha=0.08,
            label="transit",
        )
        ax.axhline(1 - case.depth_ppm * 1e-6, color="k", lw=0.6, ls=":", label="injected depth")
        ax.set_xlabel("hours from mid-transit")
        if row == 0:
            ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Detrending diagnostic on SYNTHETIC light curves: transit windows shaded red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def real_comparison(limit: int, dataset_cfg: PreprocessingConfig) -> pd.DataFrame:
    paths = get_paths()
    repo = TargetRepository(paths)
    old_cfg = PreprocessingConfig(
        detrend=DetrendConfig(method="running_median", window_days=1.5, mask_known_transits=False)
    )
    rows = []
    for kepid in repo.downloaded_kepids()[:limit]:
        kois = repo.kois(kepid)
        ephs = [e for e in (ephemeris_from_row(r) for _, r in kois.iterrows()) if e is not None]
        files = repo.product_files(kepid)
        try:
            old = preprocess_files(files, old_cfg).frame
            new = preprocess_files(files, dataset_cfg, ephemerides=ephs).frame
        except (OSError, ValueError) as exc:
            logger.warning("KIC %d skipped: %s", kepid, exc)
            continue
        for _, r in kois.iterrows():
            eph = ephemeris_from_row(r)
            if eph is None or eph.duration_hours is None:
                continue
            d_old = folded_depth_ppm(old.time_bkjd.to_numpy(), old.flux_detrended.to_numpy(), eph)
            d_new = folded_depth_ppm(new.time_bkjd.to_numpy(), new.flux_detrended.to_numpy(), eph)
            rows.append(
                {
                    "kepid": kepid,
                    "kepoi_name": r.kepoi_name,
                    "koi_disposition": r.koi_disposition,
                    "n_kois_on_star": len(kois),
                    "koi_period": eph.period_days,
                    "koi_duration_h": eph.duration_hours,
                    "koi_depth_catalog_ppm": r.koi_depth,
                    "depth_m1_running_median_ppm": d_old,
                    "depth_masked_spline_ppm": d_new,
                    "ratio_new_over_old": d_new / d_old if d_old else float("nan"),
                    "oot_scatter_old_ppm": float(np.std(old.flux_detrended) * 1e6),
                    "oot_scatter_new_ppm": float(
                        np.std(new.flux_detrended[~new.in_known_transit_mask]) * 1e6
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--real-limit",
        type=int,
        default=20,
        help="number of downloaded stars to compare (0 = skip)",
    )
    parser.add_argument("--data-config", default="configs/data/kepler_dr25_small.yaml")
    args = parser.parse_args()
    setup_logging()
    paths = get_paths()
    out = paths.reports / "detrending"
    out.mkdir(parents=True, exist_ok=True)

    raw, summary = synthetic(args.seeds)
    raw.to_csv(out / "synthetic_results.csv", index=False)
    summary.to_csv(out / "synthetic_summary.csv", index=False)
    print(
        summary.pivot(index="case", columns="method", values="mean_relative_depth_error")
        .round(4)
        .to_string()
    )
    fig_path = paths.figures / "detrending_synthetic_cases.png"
    plot_cases(fig_path)

    cfg = load_config(resolve_path(args.data_config, paths), DataConfig)
    assert cfg.dataset is not None
    real = (
        real_comparison(args.real_limit, cfg.dataset.preprocessing)
        if args.real_limit
        else pd.DataFrame()
    )
    if not real.empty:
        real.to_csv(out / "real_comparison.csv", index=False)
        print(
            real[
                [
                    "kepid",
                    "kepoi_name",
                    "koi_disposition",
                    "koi_duration_h",
                    "koi_depth_catalog_ppm",
                    "depth_m1_running_median_ppm",
                    "depth_masked_spline_ppm",
                    "ratio_new_over_old",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    masked = summary[(summary.method == "robust_spline_masked") & summary.acceptance_case]
    write_json_atomic(
        out / "summary.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data": "SYNTHETIC light curves (evaluation.detrending) + real Kepler comparison",
            "seeds": args.seeds,
            "acceptance": {
                "mean_abs_max": MEAN_TOL,
                "rms_max": RMS_TOL,
                "cases": list(ACCEPTANCE_CASES),
            },
            "dataset_method": cfg.dataset.preprocessing.detrend.model_dump(),
            "robust_spline_masked_passes_all": bool(masked.passes.all()),
            "synthetic_summary": summary.round(5).to_dict("records"),
            "real_comparison": {
                "n_kois": len(real),
                "median_ratio_new_over_old": float(real.ratio_new_over_old.median())
                if not real.empty
                else None,
                # Catalog depth is a fitted model depth, not ground truth; ratios are context.
                **(
                    {
                        "median_ratio_old_over_catalog": float(
                            (real.depth_m1_running_median_ppm / real.koi_depth_catalog_ppm).median()
                        ),
                        "median_ratio_new_over_catalog": float(
                            (real.depth_masked_spline_ppm / real.koi_depth_catalog_ppm).median()
                        ),
                        "fraction_new_deeper_than_old": float((real.ratio_new_over_old > 1).mean()),
                    }
                    if not real.empty
                    else {}
                ),
            },
            "figure": str(fig_path.relative_to(paths.root)),
        },
    )
    print(f"\nrobust_spline_masked passes acceptance on all cases: {bool(masked.passes.all())}")
    print(f"Wrote {out.relative_to(paths.root)}/ and {fig_path.relative_to(paths.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
