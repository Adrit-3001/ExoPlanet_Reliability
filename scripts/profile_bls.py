"""Profile the BLS baseline on one real processed light curve.

Measures where time goes and how runtime scales, using evenly strided sub-grids of the
production period grid (runtime is linear in the number of periods; this is verified
here) so the whole profile takes a few minutes. Nothing here changes the BLS
configuration used by scripts/run_bls.py.

Output: artifacts/reports/bls_profiling/profile.json

Example:
    python scripts/profile_bls.py --kepid 5374854
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

from exoreliability.baselines.bls import _power, period_grid, run_bls
from exoreliability.config import BLSConfig, get_paths
from exoreliability.data.cache import write_json_atomic
from exoreliability.preprocessing.pipeline import load_processed


def timed(fn, *args, repeat: int = 1, **kwargs):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kepid", type=int, default=5374854)
    parser.add_argument(
        "--stride", type=int, default=20, help="use every Nth period of the production grid"
    )
    args = parser.parse_args()
    paths = get_paths()
    processed = load_processed(paths, args.kepid)
    if processed is None:
        raise SystemExit(
            f"KIC {args.kepid} has no processed light curve (run scripts/preprocess_lightcurves.py)"
        )
    lc = processed.to_lightcurve()
    cfg = BLSConfig()
    t, y, dy = lc.time, lc.flux, lc.flux_err
    baseline = float(t.max() - t.min())
    durations = np.asarray(cfg.durations_hours) / 24.0
    grid = period_grid(
        baseline,
        cfg.min_period_days,
        min(cfg.max_period_days, baseline / (cfg.min_transits - 1)),
        float(durations.min()),
        cfg.oversample,
    )
    sub = grid[:: args.stride]
    results: dict = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "kepid": args.kepid,
        "n_points": int(t.size),
        "baseline_days": baseline,
        "full_grid_periods": int(grid.size),
        "subgrid_stride": args.stride,
        "subgrid_periods": int(sub.size),
        "cpu_count": os.cpu_count(),
        "config": cfg.model_dump(),
    }

    # 1. Linearity in number of periods (single process).
    t_half, _ = timed(_power, t, y, dy, sub[::2], durations, cfg.objective, 1)
    t_sub, pg_ref = timed(_power, t, y, dy, sub, durations, cfg.objective, 1)
    results["single_process_seconds"] = {"subgrid_half": t_half, "subgrid": t_sub}
    results["seconds_per_1000_periods_single"] = 1000 * t_sub / sub.size
    results["extrapolated_full_grid_single_seconds"] = t_sub * grid.size / sub.size

    # 2. Where the time goes (cProfile of one single-process periodogram).
    prof = cProfile.Profile()
    prof.enable()
    BoxLeastSquares(t, y, dy=dy).power(sub, durations, objective=cfg.objective)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats("tottime")
    top = []
    for (file, line, func), (_cc, nc, tt, ct, _callers) in sorted(
        stats.stats.items(), key=lambda kv: -kv[1][2]
    )[:8]:
        top.append(
            {
                "function": f"{Path(file).name}:{line}({func})",
                "calls": nc,
                "tottime_s": round(tt, 3),
                "cumtime_s": round(ct, 3),
            }
        )
    results["cprofile_top_tottime"] = top

    # 3. Scaling with number of durations (single process).
    results["durations_scaling_seconds"] = {
        str(k): timed(_power, t, y, dy, sub, durations[:k], cfg.objective, 1)[0]
        for k in (1, 3, len(durations))
    }

    # 4. Astropy's internal phase-binning 'oversample' (default 10) — changes precision,
    #    i.e. a methodology change; measured only.
    model = BoxLeastSquares(t, y, dy=dy)
    over = {}
    for o in (5, 10, 20):
        sec, pg = timed(model.power, sub, durations, objective=cfg.objective, oversample=o)
        over[str(o)] = {"seconds": sec, "best_period": float(sub[int(np.argmax(pg.power))])}
    results["astropy_oversample_seconds"] = over

    # 5. Worker scaling on the sub-grid (results must be identical).
    workers = {}
    for n in (1, 2, 4, 6, 8, 10, 12, 16):
        if n > (os.cpu_count() or 1):
            continue
        sec, pg = timed(_power, t, y, dy, sub, durations, cfg.objective, n)
        workers[str(n)] = {
            "seconds": sec,
            "speedup": t_sub / sec,
            "identical_to_single": bool(np.array_equal(pg["power"], pg_ref["power"])),
        }
    results["worker_scaling"] = workers

    # 6. compute_stats (called once per run) and a full production run for reference.
    best = int(np.argmax(pg_ref["power"]))
    sec_stats, _ = timed(
        model.compute_stats,
        float(sub[best]),
        float(pg_ref["duration"][best]),
        float(pg_ref["transit_time"][best]),
    )
    results["compute_stats_seconds"] = sec_stats
    sec_full, (res, _) = timed(run_bls, lc, cfg)
    results["full_run_seconds_default_n_jobs"] = sec_full
    results["full_run_best_period"] = res.period_days

    out = paths.reports / "bls_profiling"
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "profile.json", results)
    import json

    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
