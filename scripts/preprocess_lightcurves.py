"""Preprocess downloaded Kepler light curves and write diagnostic figures.

Reads raw FITS listed in each target's ``products.json``, runs the preprocessing chain
configured in the experiment config, writes ``data/processed/lightcurves/kic_*/`` and a
diagnostic plot to ``artifacts/figures/kic_*_diagnostic.png``.

Example:
    python scripts/preprocess_lightcurves.py                 # all downloaded targets
    python scripts/preprocess_lightcurves.py --kepid 10811496
"""

from __future__ import annotations

import argparse
import logging

from exoreliability.config import (
    BLSExperimentConfig,
    DataConfig,
    get_paths,
    load_config,
    resolve_path,
)
from exoreliability.data.cache import kic_dirname
from exoreliability.data.catalog import ephemeris_from_row
from exoreliability.logging import setup_logging
from exoreliability.plotting import plot_target_diagnostic
from exoreliability.preprocessing.pipeline import preprocess_files, save_processed
from exoreliability.targets import TargetRepository

logger = logging.getLogger("preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiments/bls_smoke.yaml")
    parser.add_argument(
        "--kepid", type=int, action="append", help="KIC id(s); default: all downloaded"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    exp = load_config(resolve_path(args.config, paths), BLSExperimentConfig)
    data_cfg = load_config(resolve_path(exp.data_config, paths), DataConfig)
    repo = TargetRepository(paths)
    kepids = args.kepid or repo.downloaded_kepids()
    if not kepids:
        logger.error("No downloaded targets; run scripts/fetch_lightcurves.py first")
        return 1

    n_ok = 0
    for kepid in kepids:
        files = repo.product_files(kepid)
        if not files:
            print(f"KIC {kepid}: no downloaded products, skipped")
            continue
        kois = repo.kois(kepid)
        ephemerides = [
            e for e in (ephemeris_from_row(r) for _, r in kois.iterrows()) if e is not None
        ]
        try:
            processed = preprocess_files(
                files,
                exp.preprocessing,
                flux_column=data_cfg.lightcurves.flux_column,
                ephemerides=ephemerides,
            )
        except (OSError, ValueError) as exc:
            logger.error("KIC %d: preprocessing failed: %s", kepid, exc)
            continue
        out_dir = save_processed(processed, paths, kepid)
        counts = processed.record["counts"]
        removed = sum(
            q["clean"]["non_finite"] + q["clean"]["quality_flagged"]
            for q in processed.record["quarters"]
        )
        releases = sorted({str(q.get("data_release")) for q in processed.record["quarters"]})
        print(
            f"KIC {kepid}: {len(files)} quarters, {counts['final']} samples kept "
            f"({removed} invalid/flagged removed, {counts['upper_outliers_clipped']} upper outliers clipped), "
            f"DATA_REL={','.join(releases)} -> {out_dir.relative_to(paths.root)}"
        )
        if not args.no_plot:
            labelled = [
                (str(r["kepoi_name"]), str(r["koi_disposition"]), eph)
                for (_, r) in kois.iterrows()
                if (eph := ephemeris_from_row(r)) is not None
            ]
            fig = plot_target_diagnostic(
                processed.frame,
                kepid=kepid,
                kois=labelled,
                out_path=paths.figures / f"{kic_dirname(kepid)}_diagnostic.png",
            )
            print(f"  figure: {fig.relative_to(paths.root)}")
        n_ok += 1
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
