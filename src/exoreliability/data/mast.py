"""Kepler light-curve retrieval from MAST (search via Lightkurve, download via astroquery).

Products are stored at deterministic paths under ``data/raw/lightcurves/kepler/kic_*/``
together with a ``products.json`` manifest. If the manifest and files are present and the
product policy is unchanged, no network request is made.

Nothing here performs network access at import time; Lightkurve/astroquery are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits

from exoreliability.config import LightCurveConfig, ProjectPaths
from exoreliability.data.cache import (
    product_manifest_path,
    raw_product_path,
    read_json,
    sha256_file,
    write_json_atomic,
)
from exoreliability.data.contracts import BKJD_OFFSET, LightCurveData

logger = logging.getLogger(__name__)

_QUARTER_RE = re.compile(r"Quarter\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ProductRecord:
    """One MAST light-curve product (one Kepler quarter for long cadence)."""

    kepid: int
    quarter: int
    filename: str
    data_uri: str
    size_bytes: int | None
    obs_id: str
    author: str
    exptime_seconds: float
    local_path: str | None = None  # relative to the project root once downloaded
    sha256: str | None = None


@dataclass
class TargetFetchResult:
    kepid: int
    status: Literal["downloaded", "cached", "no_products", "error"]
    products: list[ProductRecord] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("downloaded", "cached")


Searcher = Callable[[int, LightCurveConfig], list[ProductRecord]]
Downloader = Callable[[str, Path], None]


def policy_dict(cfg: LightCurveConfig) -> dict[str, Any]:
    """Fields of the product policy that determine which files are selected."""
    return {
        "mission": cfg.mission,
        "author": cfg.author,
        "exptime_seconds": cfg.exptime_seconds,
        "quarters": cfg.quarters,
        "max_products_per_target": cfg.max_products_per_target,
    }


def search_products(kepid: int, cfg: LightCurveConfig) -> list[ProductRecord]:
    """Search MAST for Kepler light-curve products matching the configured policy."""
    import lightkurve as lk  # heavy import, deferred

    result = lk.search_lightcurve(
        f"KIC {kepid}",
        mission=cfg.mission,
        author=cfg.author,
        exptime=cfg.exptime_seconds,
    )
    records: list[ProductRecord] = []
    if len(result) == 0:
        return records
    table = result.table
    for row in table:
        target_name = str(row["target_name"])
        # Guard against cone-search matches of neighbouring stars.
        if not target_name.startswith("kplr") or int(target_name[4:]) != kepid:
            continue
        match = _QUARTER_RE.search(str(row["mission"]))
        if match is None:
            continue
        size = row["size"]
        records.append(
            ProductRecord(
                kepid=kepid,
                quarter=int(match.group(1)),
                filename=str(row["productFilename"]),
                data_uri=str(row["dataURI"]),
                size_bytes=None if np.ma.is_masked(size) else int(size),
                obs_id=str(row["obs_id"]),
                author=str(row["author"]),
                exptime_seconds=float(row["exptime"]),
            )
        )
    return sorted(records, key=lambda r: (r.quarter, r.filename))


def download_product(data_uri: str, destination: Path) -> None:
    """Download one MAST product to ``destination`` atomically."""
    from astroquery.mast import Observations  # deferred

    # Safe only after the import: astropy has now created its own logger class for it.
    logging.getLogger("astroquery").setLevel(logging.WARNING)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    status, msg, _url = Observations.download_file(data_uri, local_path=str(tmp), cache=False)
    if status != "COMPLETE" or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise OSError(f"MAST download failed for {data_uri}: {status} {msg}")
    tmp.replace(destination)


def select_products(records: list[ProductRecord], cfg: LightCurveConfig) -> list[ProductRecord]:
    """Apply the quarter filter and per-target product cap (lowest quarters first)."""
    chosen = [r for r in records if cfg.quarters is None or r.quarter in cfg.quarters]
    # One product per quarter; duplicates would indicate an unexpected archive state.
    seen: dict[int, ProductRecord] = {}
    for r in chosen:
        seen.setdefault(r.quarter, r)
    return sorted(seen.values(), key=lambda r: r.quarter)[: cfg.max_products_per_target]


def load_cached_products(
    kepid: int, cfg: LightCurveConfig, paths: ProjectPaths
) -> list[ProductRecord] | None:
    """Return manifest products if every file is present and the policy matches."""
    manifest_path = product_manifest_path(paths, kepid)
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("policy") != policy_dict(cfg):
        logger.info("KIC %d: product policy changed; cached manifest ignored", kepid)
        return None
    records = [ProductRecord(**p) for p in manifest.get("products", [])]
    for r in records:
        if r.local_path is None or not (paths.root / r.local_path).exists():
            return None
    return records


def fetch_target(
    kepid: int,
    cfg: LightCurveConfig,
    paths: ProjectPaths,
    *,
    searcher: Searcher = search_products,
    downloader: Downloader = download_product,
    remaining_budget_bytes: float | None = None,
) -> TargetFetchResult:
    """Ensure the configured products for one KIC target are available locally.

    Never raises for per-target problems (no products, download failure); these are
    reported in the returned ``TargetFetchResult`` so batch jobs can continue.
    """
    cached = load_cached_products(kepid, cfg, paths)
    if cached is not None:
        return TargetFetchResult(kepid, "cached", cached, f"{len(cached)} cached products")

    try:
        found = searcher(kepid, cfg)
    except Exception as exc:
        logger.warning("KIC %d: MAST search failed: %s", kepid, exc)
        return TargetFetchResult(kepid, "error", message=f"search failed: {exc}")
    selected = select_products(found, cfg)
    if not selected:
        return TargetFetchResult(
            kepid, "no_products", message=f"no {cfg.author} {cfg.exptime_seconds}s products found"
        )

    to_download = [r for r in selected if not raw_product_path(paths, kepid, r.filename).exists()]
    needed = sum(r.size_bytes or 0 for r in to_download)
    if remaining_budget_bytes is not None and needed > remaining_budget_bytes:
        return TargetFetchResult(
            kepid,
            "error",
            message=f"download of {needed / 1e6:.1f} MB exceeds remaining budget "
            f"{remaining_budget_bytes / 1e6:.1f} MB",
        )

    stored: list[ProductRecord] = []
    for r in selected:
        dest = raw_product_path(paths, kepid, r.filename)
        if not dest.exists():
            logger.info("KIC %d: downloading Q%d %s", kepid, r.quarter, r.filename)
            try:
                downloader(r.data_uri, dest)
            except Exception as exc:
                logger.warning("KIC %d: download failed for %s: %s", kepid, r.filename, exc)
                return TargetFetchResult(kepid, "error", stored, f"download failed: {exc}")
        stored.append(
            ProductRecord(
                **{
                    **asdict(r),
                    "local_path": str(dest.relative_to(paths.root)),
                    "sha256": sha256_file(dest),
                }
            )
        )

    write_json_atomic(
        product_manifest_path(paths, kepid),
        {
            "kepid": kepid,
            "policy": policy_dict(cfg),
            "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "search": {
                "service": "MAST via lightkurve.search_lightcurve",
                "target": f"KIC {kepid}",
                "n_found": len(found),
            },
            "products": [asdict(r) for r in stored],
        },
    )
    return TargetFetchResult(kepid, "downloaded", stored, f"{len(stored)} products stored")


def read_kepler_fits(path: Path, flux_column: str = "PDCSAP_FLUX") -> LightCurveData:
    """Read one Kepler ``*_llc.fits`` file without any filtering.

    Returns raw columns (NaNs and flagged cadences included) so that cleaning is an
    explicit, recorded pipeline step. Time is BKJD.
    """
    with fits.open(path, memmap=False) as hdul:
        primary, lc = hdul[0].header, hdul[1]
        bjdref = float(lc.header.get("BJDREFI", 0)) + float(lc.header.get("BJDREFF", 0.0))
        if not np.isclose(bjdref, BKJD_OFFSET):
            raise ValueError(f"{path}: unexpected BJD reference {bjdref}, expected BKJD")
        data = lc.data
        time = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data[flux_column], dtype=np.float64)
        flux_err = np.asarray(data[f"{flux_column}_ERR"], dtype=np.float64)
        quality = np.asarray(data["SAP_QUALITY"], dtype=np.int64)
        quarter = int(primary["QUARTER"])
        meta = {
            "file": Path(path).name,
            "kepid": int(primary["KEPLERID"]),
            "quarter": quarter,
            "data_release": primary.get("DATA_REL"),
            "processing_version": primary.get("PROCVER"),
            "obsmode": primary.get("OBSMODE"),
            "flux_column": flux_column,
            "kepmag": primary.get("KEPMAG"),
            "channel": primary.get("CHANNEL"),
            "crowdsap": lc.header.get("CROWDSAP"),
            "flfrcsap": lc.header.get("FLFRCSAP"),
            "pdc_method": lc.header.get("PDCMETHD"),
        }
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        quarter=np.full(len(time), quarter, dtype=np.int64),
        meta=meta,
    )
