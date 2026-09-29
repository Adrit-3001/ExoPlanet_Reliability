"""Create star-level (KIC-grouped) train/val/test splits over the DR25 KOI snapshot.

Writes data/splits/{train,val,test}.csv and split_metadata.json. Existing splits are never
replaced unless --overwrite is given.

Example:
    python scripts/create_splits.py
"""

from __future__ import annotations

import argparse
import json
import logging

from exoreliability.config import SplitConfig, get_paths, load_config, resolve_path
from exoreliability.data.archive import ArchiveError, fetch_dr25_koi_catalog
from exoreliability.data.labels import apply_label_policy
from exoreliability.data.splits import SplitsExistError, build_split_manifest, write_splits
from exoreliability.logging import setup_logging

logger = logging.getLogger("create_splits")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data/splits_dr25.yaml")
    parser.add_argument("--overwrite", action="store_true", help="replace existing split files")
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    cfg = load_config(resolve_path(args.config, paths), SplitConfig)
    try:
        snapshot = fetch_dr25_koi_catalog(paths)
    except ArchiveError as exc:
        logger.error("Catalog unavailable: %s", exc)
        return 1
    manifest = build_split_manifest(apply_label_policy(snapshot.table), cfg)
    try:
        meta = write_splits(
            manifest, paths.splits, cfg=cfg, catalog_meta=snapshot.meta, overwrite=args.overwrite
        )
    except SplitsExistError as exc:
        logger.error("%s", exc)
        return 1
    print(
        json.dumps({"splits": meta["splits"], "leakage_checks": meta["leakage_checks"]}, indent=2)
    )
    print(
        f"\nWrote {paths.splits.relative_to(paths.root)}/{{train,val,test}}.csv + split_metadata.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
