"""NASA Exoplanet Archive access via the Table Access Protocol (TAP) service.

All archive HTTP traffic in this project goes through this module. Nothing here touches
the network at import time.

TAP documentation: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from exoreliability.config import ProjectPaths
from exoreliability.data.cache import (
    catalog_meta_path,
    catalog_snapshot_path,
    read_json,
    sha256_file,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DR25_KOI_TABLE = "q1_q17_dr25_koi"

# Columns verified against TAP_SCHEMA.columns for Q1_Q17_DR25_KOI (see docs/data_contract.md).
DR25_KOI_COLUMNS: tuple[str, ...] = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_disp_prov",
    "koi_vet_stat",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_period_err1",
    "koi_period_err2",
    "koi_time0bk",
    "koi_time0bk_err1",
    "koi_time0bk_err2",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_impact",
    "koi_model_snr",
    "koi_num_transits",
    "koi_prad",
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "koi_quarters",
    "koi_tce_plnt_num",
    "koi_count",
    "ra",
    "dec",
)


class ArchiveError(RuntimeError):
    """Raised when the archive returns an error or an unparseable response."""


def build_select_query(
    table: str,
    columns: tuple[str, ...] | list[str],
    where: str | None = None,
    order_by: str | None = None,
) -> str:
    """Build an ADQL SELECT statement. Identifiers are validated, not escaped."""
    for ident in (table, *columns):
        if not ident.replace("_", "").isalnum():
            raise ValueError(f"invalid ADQL identifier: {ident!r}")
    query = f"select {','.join(columns)} from {table}"
    if where:
        query += f" where {where}"
    if order_by:
        query += f" order by {order_by}"
    return query


def run_tap_query(
    query: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Execute a synchronous TAP query and return the CSV result as a DataFrame."""
    params = {"query": query, "format": "csv"}
    own_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = http.get(TAP_SYNC_URL, params=params)
    except httpx.HTTPError as exc:
        raise ArchiveError(f"TAP request failed: {exc}") from exc
    finally:
        if own_client:
            http.close()
    if response.status_code != 200:
        raise ArchiveError(f"TAP returned HTTP {response.status_code}: {response.text[:500]}")
    text = response.text
    # TAP errors are sometimes returned as a VOTable/XML body with HTTP 200.
    if text.lstrip().startswith("<"):
        raise ArchiveError(f"TAP returned a non-CSV response: {text[:500]}")
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ArchiveError(f"could not parse TAP CSV: {exc}") from exc


@dataclass(frozen=True)
class CatalogSnapshot:
    """A cached catalog table plus the metadata needed to reproduce it."""

    table: pd.DataFrame
    meta: dict[str, Any]
    from_cache: bool


def fetch_dr25_koi_catalog(
    paths: ProjectPaths,
    *,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> CatalogSnapshot:
    """Return the full DR25 KOI table (selected columns), downloading it once.

    The full table is ~8k rows, so the snapshot is fetched whole and subsets are drawn
    locally; this keeps subset selection reproducible from a single recorded query.
    """
    csv_path = catalog_snapshot_path(paths, DR25_KOI_TABLE)
    meta_path = catalog_meta_path(paths, DR25_KOI_TABLE)
    if csv_path.exists() and meta_path.exists() and not refresh:
        meta = read_json(meta_path)
        if meta.get("sha256") != sha256_file(csv_path):
            raise ArchiveError(f"{csv_path} does not match the checksum in {meta_path}")
        logger.info(
            "Using cached catalog snapshot %s (retrieved %s)", csv_path, meta["retrieved_at_utc"]
        )
        return CatalogSnapshot(pd.read_csv(csv_path), meta, from_cache=True)

    query = build_select_query(DR25_KOI_TABLE, DR25_KOI_COLUMNS, order_by="kepoi_name")
    logger.info("Querying NASA Exoplanet Archive TAP: %s", query)
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    table = run_tap_query(query, client=client)
    missing = set(DR25_KOI_COLUMNS) - set(table.columns)
    if missing:
        raise ArchiveError(f"archive response missing expected columns: {sorted(missing)}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    meta = {
        "source": "NASA Exoplanet Archive",
        "service": "TAP sync",
        "endpoint": TAP_SYNC_URL,
        "table": DR25_KOI_TABLE,
        "query": query,
        "format": "csv",
        "retrieved_at_utc": retrieved_at,
        "n_rows": len(table),
        "n_unique_kepid": int(table["kepid"].nunique()),
        "columns": list(table.columns),
        "sha256": sha256_file(csv_path),
    }
    write_json_atomic(meta_path, meta)
    return CatalogSnapshot(table, meta, from_cache=False)
