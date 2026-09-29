"""Fetch the Kepler Q1-Q17 DR25 KOI table and write a small, reproducible KOI manifest.

Example:
    python scripts/fetch_catalog.py --limit 20
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from exoreliability.config import DataConfig, SelectionConfig, get_paths, load_config, resolve_path
from exoreliability.data.archive import ArchiveError, fetch_dr25_koi_catalog
from exoreliability.data.catalog import assign_labels, select_subset, write_koi_manifest
from exoreliability.logging import setup_logging

logger = logging.getLogger("fetch_catalog")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data/kepler_dr25_small.yaml")
    parser.add_argument(
        "--limit", type=int, help="number of KOIs in the manifest (overrides config)"
    )
    parser.add_argument("--seed", type=int, help="selection seed (overrides config)")
    parser.add_argument("--refresh", action="store_true", help="re-download the catalog snapshot")
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    config_path = resolve_path(args.config, paths)
    cfg = load_config(config_path, DataConfig)
    overrides = {k: v for k, v in {"limit": args.limit, "seed": args.seed}.items() if v is not None}
    selection = SelectionConfig.model_validate({**cfg.selection.model_dump(), **overrides})

    try:
        snapshot = fetch_dr25_koi_catalog(paths, refresh=args.refresh)
    except ArchiveError as exc:
        logger.error("Catalog retrieval failed: %s", exc)
        return 1

    labelled = assign_labels(snapshot.table)
    meta = snapshot.meta
    print(f"\nSource: {meta['source']} {meta['service']} table={meta['table']}")
    print(f"Query:  {meta['query']}")
    print(
        f"Retrieved: {meta['retrieved_at_utc']}  ({'cache' if snapshot.from_cache else 'network'})"
    )
    print(
        f"Rows: {meta['n_rows']} KOIs on {meta['n_unique_kepid']} stars  sha256={meta['sha256'][:12]}…\n"
    )
    print("koi_disposition × koi_pdisposition:")
    print(
        pd.crosstab(
            labelled["koi_disposition"], labelled["koi_pdisposition"], dropna=False
        ).to_string()
    )
    print("\nDerived labels (policy dr25_clean_v2):")
    print(labelled["label_name"].replace("", "excluded").value_counts().to_string())
    print("\nExclusion reasons:")
    print(labelled.loc[labelled["label"].isna(), "exclusion_reason"].value_counts().to_string())

    subset = select_subset(labelled, selection)
    manifest = write_koi_manifest(
        subset, name=cfg.name, paths=paths, catalog_meta=meta, selection=selection
    )
    cols = [
        "kepid",
        "kepoi_name",
        "koi_disposition",
        "koi_pdisposition",
        "label_name",
        "koi_period",
        "koi_depth",
        "koi_duration",
        "koi_model_snr",
        "koi_kepmag",
    ]
    print(
        f"\nSelected {len(subset)} KOIs on {subset['kepid'].nunique()} stars (seed={selection.seed}):"
    )
    print(subset[cols].to_string(index=False))
    print(
        f"\nManifest: {manifest.relative_to(paths.root)} (+ {manifest.with_suffix('.json').name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
