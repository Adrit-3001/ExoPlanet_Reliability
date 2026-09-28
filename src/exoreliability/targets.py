"""Read-only access to locally available per-target data (catalog, light curves, BLS runs).

This is the service layer behind the API. It never touches the network: targets must be
fetched and processed by the scripts first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from exoreliability.config import ProjectPaths
from exoreliability.data.archive import DR25_KOI_TABLE
from exoreliability.data.cache import (
    catalog_meta_path,
    catalog_snapshot_path,
    product_manifest_path,
    read_json,
)
from exoreliability.data.catalog import assign_labels, kois_for_target
from exoreliability.preprocessing.pipeline import ProcessedLightCurve, load_processed

_TARGET_ID_RE = re.compile(r"^(?:kic[\s_-]?)?0*(\d{1,9})$", re.IGNORECASE)


def parse_target_id(target_id: str) -> int:
    """Accept ``10811496``, ``KIC 10811496``, ``kic_010811496``; raise ``ValueError`` otherwise."""
    match = _TARGET_ID_RE.match(target_id.strip())
    if match is None:
        raise ValueError(f"not a KIC identifier: {target_id!r}")
    return int(match.group(1))


@dataclass
class TargetRepository:
    paths: ProjectPaths

    def __post_init__(self) -> None:
        self._catalog: pd.DataFrame | None = None
        self._catalog_mtime: float | None = None

    # -- catalog ---------------------------------------------------------------
    def catalog(self) -> pd.DataFrame | None:
        path = catalog_snapshot_path(self.paths, DR25_KOI_TABLE)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if self._catalog is None or mtime != self._catalog_mtime:
            self._catalog = assign_labels(pd.read_csv(path))
            self._catalog_mtime = mtime
        return self._catalog

    def catalog_meta(self) -> dict[str, Any] | None:
        path = catalog_meta_path(self.paths, DR25_KOI_TABLE)
        return read_json(path) if path.exists() else None

    def kois(self, kepid: int) -> pd.DataFrame:
        cat = self.catalog()
        return pd.DataFrame() if cat is None else kois_for_target(cat, kepid)

    # -- light curves ------------------------------------------------------------
    def product_manifest(self, kepid: int) -> dict[str, Any] | None:
        path = product_manifest_path(self.paths, kepid)
        return read_json(path) if path.exists() else None

    def product_files(self, kepid: int) -> list[Path]:
        manifest = self.product_manifest(kepid)
        if manifest is None:
            return []
        return [
            self.paths.root / p["local_path"] for p in manifest["products"] if p.get("local_path")
        ]

    def downloaded_kepids(self) -> list[int]:
        root = self.paths.raw_lightcurves / "kepler"
        if not root.exists():
            return []
        return sorted(int(p.parent.name.split("_")[1]) for p in root.glob("kic_*/products.json"))

    def processed(self, kepid: int) -> ProcessedLightCurve | None:
        return load_processed(self.paths, kepid)

    def processed_kepids(self) -> list[int]:
        root = self.paths.processed_lightcurves
        if not root.exists():
            return []
        return sorted(
            int(p.parent.name.split("_")[1]) for p in root.glob("kic_*/lightcurve.parquet")
        )

    # -- BLS runs ------------------------------------------------------------------
    def latest_bls(self, kepid: int) -> tuple[Path, dict[str, Any]] | None:
        """Most recent BLS run folder containing a successful entry for ``kepid``."""
        if not self.paths.experiments.exists():
            return None
        for run_dir in sorted(self.paths.experiments.iterdir(), reverse=True):
            results = run_dir / "results.json"
            if not results.exists():
                continue
            doc = read_json(results)
            if doc.get("kind") != "bls":
                continue
            for entry in doc.get("targets", []):
                if entry.get("kepid") == kepid and entry.get("status") == "ok":
                    return run_dir, {"run_id": doc["run_id"], **entry}
        return None
