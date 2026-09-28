"""Blind BLS search on preprocessed Kepler light curves, with an optional noise demonstration.

For each target: run BLS on the detrended light curve, compare the best period with every
catalog KOI period on that star, then (optionally) repeat the search after adding seeded
Gaussian noise at the configured severities. Results go to a new folder under
``artifacts/experiments/`` (config.yaml, environment.json, results.json, summary.csv,
periodograms/, figures/).

This validates the pipeline on individual objects; it is not a performance evaluation.

Example:
    python scripts/run_bls.py --kepid 10811496
    python scripts/run_bls.py --no-noise-demo          # all processed targets, clean only
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from exoreliability.baselines.bls import BLSResult, compare_periods, downsample_periodogram, run_bls
from exoreliability.config import BLSExperimentConfig, get_paths, load_config, resolve_path
from exoreliability.data.cache import kic_dirname, write_json_atomic
from exoreliability.experiments.results import create_run_dir, write_run_metadata
from exoreliability.logging import setup_logging
from exoreliability.perturbations.gaussian_noise import GaussianNoise
from exoreliability.plotting import plot_bls_periodograms
from exoreliability.targets import TargetRepository
from exoreliability.training.reproducibility import set_seeds

logger = logging.getLogger("run_bls")


def _catalog_rows(kois: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, r in kois.iterrows():
        rows.append(
            {
                "kepoi_name": r["kepoi_name"],
                "koi_disposition": r["koi_disposition"],
                "koi_pdisposition": r["koi_pdisposition"],
                "label_name": r["label_name"] or None,
                "koi_period": None if pd.isna(r["koi_period"]) else float(r["koi_period"]),
                "koi_time0bk": None if pd.isna(r["koi_time0bk"]) else float(r["koi_time0bk"]),
                "koi_duration": None if pd.isna(r["koi_duration"]) else float(r["koi_duration"]),
                "koi_depth": None if pd.isna(r["koi_depth"]) else float(r["koi_depth"]),
            }
        )
    return rows


def _compare(result: BLSResult, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for koi in catalog:
        if koi["koi_period"]:
            out.append(
                {
                    "kepoi_name": koi["kepoi_name"],
                    "catalog_period": koi["koi_period"],
                    **compare_periods(result.period_days, koi["koi_period"]),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiments/bls_smoke.yaml")
    parser.add_argument(
        "--kepid", type=int, action="append", help="KIC id(s); default: all processed"
    )
    parser.add_argument("--no-noise-demo", action="store_true")
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    config_path = resolve_path(args.config, paths)
    cfg = load_config(config_path, BLSExperimentConfig)
    set_seeds(cfg.seed)
    repo = TargetRepository(paths)
    kepids = args.kepid or repo.processed_kepids()
    if not kepids:
        logger.error("No processed targets; run scripts/preprocess_lightcurves.py first")
        return 1
    severities = [0.0]
    if cfg.noise_demo.enabled and not args.no_noise_demo:
        severities = sorted(set([0.0, *cfg.noise_demo.severities]))

    run_dir = create_run_dir(paths, cfg.name)
    write_run_metadata(
        run_dir,
        config=cfg.model_dump(),
        config_source=config_path,
        paths=paths,
        extra={"catalog_snapshot": repo.catalog_meta()},
    )
    (run_dir / "periodograms").mkdir()
    noise = GaussianNoise()
    targets: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for kepid in kepids:
        processed = repo.processed(kepid)
        if processed is None:
            targets.append({"kepid": kepid, "status": "missing", "message": "not processed"})
            continue
        clean_lc = processed.to_lightcurve(detrended=True)
        catalog = _catalog_rows(repo.kois(kepid))
        entry: dict[str, Any] = {
            "kepid": kepid,
            "status": "ok",
            "preprocessing_created_at_utc": processed.record.get("created_at_utc"),
            "input_sha256": [i["sha256"] for i in processed.record.get("inputs", [])],
            "kois": catalog,
            "runs": [],
        }
        plots = []
        for severity in severities:
            seed_entropy = [cfg.noise_demo.seed, kepid, round(severity * 1000)]
            rng = np.random.default_rng(seed_entropy)
            perturbed = noise.apply(clean_lc, severity, rng)
            started = time.perf_counter()
            try:
                result, pg = run_bls(perturbed.light_curve, cfg.bls)
            except ValueError as exc:
                logger.error("KIC %d severity %.2f: BLS failed: %s", kepid, severity, exc)
                entry["runs"].append({"severity": severity, "status": "error", "message": str(exc)})
                continue
            elapsed = time.perf_counter() - started
            comparison = _compare(result, catalog)
            pg_file = run_dir / "periodograms" / f"{kic_dirname(kepid)}_sev{severity:g}.json"
            write_json_atomic(pg_file, downsample_periodogram(pg["period"], pg["power"]))
            entry["runs"].append(
                {
                    "status": "ok",
                    "perturbation": perturbed.params["perturbation"],
                    "severity": severity,
                    "rng_seed_entropy": seed_entropy,
                    "perturbation_params": perturbed.params,
                    "runtime_seconds": round(elapsed, 2),
                    "bls": result.to_dict(),
                    "catalog_comparison": comparison,
                    "periodogram_file": str(pg_file.relative_to(run_dir)),
                }
            )
            best = min(comparison, key=lambda c: c["relative_error"], default=None)
            summary_rows.append(
                {
                    "kepid": kepid,
                    "severity": severity,
                    "sigma_added": perturbed.params["sigma_added"],
                    "bls_period": result.period_days,
                    "bls_duration_hours": result.duration_hours,
                    "bls_depth_ppm": result.depth * 1e6,
                    "bls_depth_snr": result.depth_snr,
                    "bls_sde": result.sde,
                    "closest_koi": best["kepoi_name"] if best else None,
                    "closest_catalog_period": best["catalog_period"] if best else None,
                    "relation": best["relation"] if best else None,
                    "harmonic": best["harmonic"] if best else None,
                }
            )
            plots.append(
                (
                    f"severity {severity:g} (σ_added={perturbed.params['sigma_added'] * 1e6:.0f} ppm)",
                    pg["period"],
                    pg["power"],
                    result.period_days,
                )
            )
            rel = best["relation"] if best else "n/a"
            print(
                f"KIC {kepid} sev={severity:<4g} P_bls={result.period_days:.5f} d  "
                f"dur={result.duration_hours:.1f} h  depth={result.depth * 1e6:.0f} ppm  "
                f"SNR={result.depth_snr:.1f}  SDE={result.sde:.1f}  vs catalog: {rel}"
                f"{' ' + best['harmonic'] if best and best['harmonic'] else ''}  ({elapsed:.0f}s)"
            )
        targets.append(entry)
        if plots:
            plot_bls_periodograms(
                plots,
                title=f"KIC {kepid} — BLS periodograms (Gaussian-noise demonstration, single target)",
                catalog_periods=[
                    (k["kepoi_name"], k["koi_period"]) for k in catalog if k["koi_period"]
                ],
                out_path=run_dir / "figures" / f"{kic_dirname(kepid)}_bls_periodograms.png",
            )

    write_json_atomic(
        run_dir / "results.json",
        {
            "kind": "bls",
            "run_id": run_dir.name,
            "note": "Pipeline validation on individual targets; not a statistical evaluation.",
            "targets": targets,
        },
    )
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    print(f"\nRun folder: {run_dir.relative_to(paths.root)}")
    return 0 if summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
