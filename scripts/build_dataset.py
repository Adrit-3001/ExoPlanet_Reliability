"""Build a fixed-length, KOI-level ML dataset from real Kepler light curves.

Steps: load the DR25 snapshot and label policy → load the persisted KIC-grouped splits
(and check they were made from the same snapshot) → select stars → fetch missing light
curves (budgeted, optional) → per star: masked robust-spline detrending, then one global
and one local phase-folded view per KOI → write data/processed/datasets/<name>/.

Per-star or per-KOI problems are recorded (build_log.csv, example_status) and never abort
the build.

Examples:
    python scripts/build_dataset.py --config configs/data/kepler_dr25_smoke.yaml
    python scripts/build_dataset.py --config configs/data/kepler_dr25_small.yaml --overwrite
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from exoreliability.config import DataConfig, get_paths, load_config, resolve_path
from exoreliability.data.archive import ArchiveError, fetch_dr25_koi_catalog
from exoreliability.data.cache import sha256_file
from exoreliability.data.examples import StarResult, build_star, write_dataset
from exoreliability.data.labels import apply_label_policy
from exoreliability.data.mast import fetch_target
from exoreliability.data.selection import select_stars
from exoreliability.data.splits import METADATA_FILE, load_splits
from exoreliability.logging import setup_logging
from exoreliability.targets import TargetRepository

logger = logging.getLogger("build_dataset")
LARGE_BUILD_STARS = 300


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data/kepler_dr25_small.yaml")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing dataset directory"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="use only already-cached light curves"
    )
    parser.add_argument(
        "--allow-large", action="store_true", help=f"allow builds with > {LARGE_BUILD_STARS} stars"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel star builds (does not change results)"
    )
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    config_path = resolve_path(args.config, paths)
    cfg = load_config(config_path, DataConfig)
    if cfg.dataset is None:
        logger.error("%s has no 'dataset' section", config_path)
        return 1
    ds = cfg.dataset
    out_dir = paths.datasets / cfg.name
    if out_dir.exists() and not args.overwrite:
        logger.error("%s exists; pass --overwrite to rebuild", out_dir)
        return 1

    try:
        snapshot = fetch_dr25_koi_catalog(paths)
    except ArchiveError as exc:
        logger.error("Catalog unavailable: %s", exc)
        return 1
    labelled = apply_label_policy(snapshot.table)
    splits_dir = resolve_path(ds.splits_dir, paths)
    splits = load_splits(splits_dir)
    if splits.metadata["source_catalog"]["sha256"] != snapshot.meta["sha256"]:
        logger.error(
            "Splits were created from a different catalog snapshot; regenerate splits explicitly"
        )
        return 1
    check = labelled.merge(
        splits.manifest[["kepoi_name", "label", "split"]], on="kepoi_name", suffixes=("", "_split")
    )
    if len(check) != len(labelled) or not check["label"].equals(check["label_split"]):
        logger.error("Labels in the split manifest disagree with the current label policy")
        return 1
    split_of = splits.split_of()

    stars = select_stars(splits, ds.stars)
    if len(stars) > LARGE_BUILD_STARS and not args.allow_large:
        logger.error(
            "%d stars selected (> %d); pass --allow-large for large builds",
            len(stars),
            LARGE_BUILD_STARS,
        )
        return 1
    print(
        f"{cfg.name} ({ds.scale}): {len(stars)} stars selected; by split: {stars.split.value_counts().to_dict()}"
    )

    repo = TargetRepository(paths)
    budget = cfg.lightcurves.max_total_download_mb * 1e6
    fetch_failures: dict[int, str] = {}
    if ds.allow_download and not args.no_download:
        for kepid in stars.kepid:
            res = fetch_target(int(kepid), cfg.lightcurves, paths, remaining_budget_bytes=budget)
            if res.status == "downloaded":
                budget -= sum(p.size_bytes or 0 for p in res.products)
            if not res.ok:
                fetch_failures[int(kepid)] = f"{res.status}: {res.message}"

    tmp_dir = out_dir.with_name(out_dir.name + ".building")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    jobs = []
    for kepid in stars.kepid.astype(int):
        kois = labelled[labelled.kepid == kepid].sort_values("kepoi_name")
        jobs.append((kepid, kois, split_of[kepid], repo.product_files(kepid)))

    started = time.perf_counter()
    results: list[StarResult] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                build_star,
                k,
                kois,
                split,
                files,
                ds,
                flux_column=cfg.lightcurves.flux_column,
                lightcurve_dir=tmp_dir / "lightcurves",
            )
            for k, kois, split, files in jobs
        ]
        for (kepid, *_), fut in zip(jobs, futures, strict=True):
            res = fut.result()
            if res.status == "no_lightcurve" and kepid in fetch_failures:
                res.message = fetch_failures[kepid]
            results.append(res)
    info = {
        "name": cfg.name,
        "scale": ds.scale,
        "data_config": str(config_path.relative_to(paths.root)),
        "source_catalog": {
            k: snapshot.meta.get(k)
            for k in ("table", "query", "retrieved_at_utc", "sha256", "n_rows")
        },
        "splits": {
            "dir": str(splits_dir.relative_to(paths.root)),
            "name": splits.metadata["name"],
            "metadata_sha256": sha256_file(splits_dir / METADATA_FILE),
        },
        "selection": {
            "n_stars": len(stars),
            "by_reason": stars.selection_reason.value_counts().to_dict(),
        },
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    doc = write_dataset(tmp_dir, results, cfg=ds, info=info)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)

    c = doc["counts"]
    print(
        f"Examples: {c['n_examples']} ({c['n_examples_ok']} ok, {c['n_trainable']} trainable); stars ok {c['n_stars_ok']}/{c['n_stars_attempted']}"
    )
    print(f"Example status: {c['example_status']}; star status: {c['star_status']}")
    for s, v in c["trainable_by_split"].items():
        print(
            f"  {s:5s}: {v['n']:4d} trainable ({v['n_positive']} pos / {v['n_negative']} neg) on {v['n_kics']} stars"
        )
    print(f"Built in {info['build_seconds']} s -> {Path(out_dir).relative_to(paths.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
