"""Deterministic on-disk locations for downloaded and derived data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from exoreliability.config import ProjectPaths


def kic_dirname(kepid: int) -> str:
    """``kic_010811496`` — zero-padded to 9 digits, matching MAST ``kplr`` file naming."""
    if kepid < 0:
        raise ValueError(f"invalid KIC id {kepid}")
    return f"kic_{kepid:09d}"


def catalog_snapshot_path(paths: ProjectPaths, table: str) -> Path:
    return paths.raw_catalogs / f"{table}.csv"


def catalog_meta_path(paths: ProjectPaths, table: str) -> Path:
    return paths.raw_catalogs / f"{table}.meta.json"


def raw_target_dir(paths: ProjectPaths, kepid: int) -> Path:
    return paths.raw_lightcurves / "kepler" / kic_dirname(kepid)


def raw_product_path(paths: ProjectPaths, kepid: int, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError(f"product filename must not contain directories: {filename!r}")
    return raw_target_dir(paths, kepid) / filename


def product_manifest_path(paths: ProjectPaths, kepid: int) -> Path:
    return raw_target_dir(paths, kepid) / "products.json"


def processed_target_dir(paths: ProjectPaths, kepid: int) -> Path:
    return paths.processed_lightcurves / kic_dirname(kepid)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a temporary file + rename so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
