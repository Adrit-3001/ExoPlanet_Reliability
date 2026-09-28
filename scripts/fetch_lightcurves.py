"""Download Kepler long-cadence light curves from MAST for a few targets in the KOI manifest.

Targets are taken in manifest order (unique KIC ids). Already-downloaded products are
reused without network access. Downloads are capped by ``max_total_download_mb``.

Examples:
    python scripts/fetch_lightcurves.py --limit 3
    python scripts/fetch_lightcurves.py --kepid 10811496 --quarters 1 2 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from exoreliability.config import DataConfig, LightCurveConfig, get_paths, load_config, resolve_path
from exoreliability.data.mast import fetch_target
from exoreliability.logging import setup_logging

logger = logging.getLogger("fetch_lightcurves")

HARD_TARGET_CAP = 25  # small-mode safety limit; bypass with --allow-large


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data/kepler_dr25_small.yaml")
    parser.add_argument(
        "--manifest", help="KOI manifest CSV (default: data/interim/manifests/<name>_kois.csv)"
    )
    parser.add_argument(
        "--limit", type=int, help="number of stars (overrides lightcurves.max_targets)"
    )
    parser.add_argument(
        "--kepid", type=int, action="append", help="explicit KIC id(s) instead of the manifest"
    )
    parser.add_argument("--quarters", type=int, nargs="+", help="restrict to these quarters")
    parser.add_argument(
        "--allow-large", action="store_true", help=f"allow more than {HARD_TARGET_CAP} targets"
    )
    args = parser.parse_args()
    setup_logging()

    paths = get_paths()
    cfg = load_config(resolve_path(args.config, paths), DataConfig)
    overrides: dict[str, object] = {}
    if args.limit is not None:
        overrides["max_targets"] = args.limit
    if args.quarters is not None:
        overrides["quarters"] = args.quarters
    lc_cfg = LightCurveConfig.model_validate({**cfg.lightcurves.model_dump(), **overrides})

    if args.kepid:
        kepids = list(dict.fromkeys(args.kepid))
    else:
        manifest_path = (
            resolve_path(args.manifest, paths)
            if args.manifest
            else paths.interim / "manifests" / f"{cfg.name}_kois.csv"
        )
        if not manifest_path.exists():
            logger.error("Manifest %s not found; run scripts/fetch_catalog.py first", manifest_path)
            return 1
        kepids = list(dict.fromkeys(pd.read_csv(manifest_path)["kepid"].astype(int)))
    kepids = kepids[: lc_cfg.max_targets]
    if len(kepids) > HARD_TARGET_CAP and not args.allow_large:
        logger.error(
            "Refusing to fetch %d targets (> %d) without --allow-large",
            len(kepids),
            HARD_TARGET_CAP,
        )
        return 1

    budget = lc_cfg.max_total_download_mb * 1e6
    rows = []
    for kepid in kepids:
        result = fetch_target(kepid, lc_cfg, paths, remaining_budget_bytes=budget)
        if result.status == "downloaded":
            budget -= sum(p.size_bytes or 0 for p in result.products)
        size_mb = sum(p.size_bytes or 0 for p in result.products) / 1e6
        quarters = [p.quarter for p in result.products]
        rows.append(
            {
                "kepid": kepid,
                "status": result.status,
                "n_products": len(result.products),
                "quarters": " ".join(map(str, quarters)),
                "size_mb": round(size_mb, 2),
                "message": result.message,
            }
        )
        print(
            f"KIC {kepid:>9}: {result.status:<11} {len(quarters):>2} quarters {size_mb:6.1f} MB  {result.message}"
        )

    summary = pd.DataFrame(rows)
    out = paths.interim / "manifests" / f"{cfg.name}_lightcurves.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # merge with earlier fetches, newest status wins
        summary = pd.concat([pd.read_csv(out), summary]).drop_duplicates("kepid", keep="last")
    summary.to_csv(out, index=False)
    n_ok = sum(r["status"] in ("downloaded", "cached") for r in rows)
    print(
        f"\n{n_ok}/{len(rows)} targets available locally. Summary: {Path(out).relative_to(paths.root)}"
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
