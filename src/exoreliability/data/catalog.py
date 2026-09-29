"""Catalog-level logic: label mapping, subset selection, and KOI lookup.

Labels are defined in ``exoreliability.data.labels`` (documented in ``docs/data_contract.md``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from exoreliability.config import ProjectPaths, SelectionConfig
from exoreliability.data.cache import sha256_file, write_json_atomic
from exoreliability.data.contracts import Ephemeris
from exoreliability.data.labels import LABEL_POLICY, NEGATIVE, POSITIVE, apply_label_policy

# Label policy lives in ``exoreliability.data.labels``; re-exported here for callers.
LABEL_RULE = LABEL_POLICY


def assign_labels(catalog: pd.DataFrame) -> pd.DataFrame:
    """Apply the current label policy (``dr25_clean_v2``); unknown values raise."""
    return apply_label_policy(catalog)


def select_subset(labelled: pd.DataFrame, selection: SelectionConfig) -> pd.DataFrame:
    """Deterministically sample ``selection.limit`` KOIs from a labelled catalog.

    Rows are first sorted by ``kepoi_name`` so the result depends only on the catalog
    content and the seed, not on the row order returned by the archive.
    """
    pool = labelled.sort_values("kepoi_name", kind="stable").reset_index(drop=True)
    if not selection.include_excluded:
        pool = pool[pool["label"].notna()]
    if len(pool) == 0:
        raise ValueError("no eligible KOIs to select from")
    limit = min(selection.limit, len(pool))

    if selection.stratify_by_label and not selection.include_excluded:
        groups = [pool[pool["label"] == label] for label in (POSITIVE, NEGATIVE)]
        per_group = _allocate(limit, [len(g) for g in groups])
        parts = [
            g.sample(n=n, random_state=selection.seed)
            for g, n in zip(groups, per_group, strict=True)
        ]
        subset = pd.concat(parts)
    else:
        subset = pool.sample(n=limit, random_state=selection.seed)
    return subset.sort_values("kepoi_name", kind="stable").reset_index(drop=True)


def _allocate(total: int, sizes: list[int]) -> list[int]:
    """Split ``total`` as evenly as possible across groups without exceeding group sizes."""
    alloc = [0] * len(sizes)
    remaining = total
    open_groups = [i for i, s in enumerate(sizes) if s > 0]
    while remaining > 0 and open_groups:
        share = max(1, math.floor(remaining / len(open_groups)))
        for i in list(open_groups):
            take = min(share, sizes[i] - alloc[i], remaining)
            alloc[i] += take
            remaining -= take
            if alloc[i] >= sizes[i]:
                open_groups.remove(i)
            if remaining == 0:
                break
    return alloc


def write_koi_manifest(
    subset: pd.DataFrame,
    *,
    name: str,
    paths: ProjectPaths,
    catalog_meta: dict[str, Any],
    selection: SelectionConfig,
) -> Path:
    """Persist a KOI subset manifest (CSV) plus provenance JSON under ``data/interim``."""
    out_dir = paths.interim / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_kois.csv"
    subset.to_csv(csv_path, index=False)
    write_json_atomic(
        csv_path.with_suffix(".json"),
        {
            "manifest": csv_path.name,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "label_rule": LABEL_RULE,
            "selection": selection.model_dump(),
            "n_kois": len(subset),
            "n_targets": int(subset["kepid"].nunique()),
            "label_counts": {
                str(k): int(v)
                for k, v in subset["label_name"].replace("", "excluded").value_counts().items()
            },
            "catalog": {
                k: catalog_meta.get(k)
                for k in ("table", "query", "retrieved_at_utc", "sha256", "n_rows")
            },
            "sha256": sha256_file(csv_path),
        },
    )
    return csv_path


def kois_for_target(catalog: pd.DataFrame, kepid: int) -> pd.DataFrame:
    return catalog[catalog["kepid"] == kepid].sort_values("kepoi_name").reset_index(drop=True)


def ephemeris_from_row(row: pd.Series) -> Ephemeris | None:
    """Catalog ephemeris (BKJD) for one KOI, or ``None`` if period/epoch are missing."""
    period, epoch = row.get("koi_period"), row.get("koi_time0bk")
    if period is None or epoch is None or pd.isna(period) or pd.isna(epoch) or float(period) <= 0:
        return None
    duration = row.get("koi_duration")
    return Ephemeris(
        period_days=float(period),
        epoch_bkjd=float(epoch),
        duration_hours=None if duration is None or pd.isna(duration) else float(duration),
    )
