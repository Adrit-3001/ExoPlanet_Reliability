"""Leakage-safe train/val/test splits grouped by star (KIC ID).

Invariant: a KIC ID appears in exactly one split, so every KOI on a star (including
excluded candidates) shares that star's split.

Algorithm ``kic_stratified_shuffle_v1``:

1. Label the catalog (``data.labels``) and give each star a stratum from its training-
   eligible KOIs: ``pos_only``, ``neg_only``, ``mixed`` or ``unlabelled_only``.
2. Within each stratum, sort KIC IDs, permute them with
   ``numpy.random.default_rng([seed, stratum_index])``, and cut them into train/val/test
   using the configured fractions (largest-remainder rounding).

Stratifying stars (not KOIs) keeps class balance close across splits while never splitting
a star. Realised KOI-level fractions differ slightly from the targets because stars carry
different numbers of KOIs; the metadata records the actual counts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoreliability.config import SplitConfig
from exoreliability.data.cache import read_json, sha256_file, write_json_atomic
from exoreliability.data.labels import LABEL_POLICY, NEGATIVE, POSITIVE

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
STRATA = ("pos_only", "neg_only", "mixed", "unlabelled_only")
ALGORITHM = "kic_stratified_shuffle_v1"
METADATA_FILE = "split_metadata.json"

MANIFEST_COLUMNS = (
    "kepid",
    "kepoi_name",
    "split",
    "star_stratum",
    "label",
    "label_name",
    "training_status",
    "exclusion_reason",
    "label_policy",
    "koi_disposition",
    "koi_pdisposition",
)


class SplitsExistError(FileExistsError):
    """Split manifests already exist and overwrite was not requested."""


class LeakageError(AssertionError):
    """A KIC ID (or KOI) appears in more than one split."""


def star_strata(labelled: pd.DataFrame) -> pd.DataFrame:
    """One row per KIC with counts of positive/negative/excluded KOIs and its stratum."""
    g = labelled.groupby("kepid", sort=True)
    stars = pd.DataFrame(
        {
            "n_pos": g["label"].apply(lambda s: int((s == POSITIVE).sum())),
            "n_neg": g["label"].apply(lambda s: int((s == NEGATIVE).sum())),
            "n_kois": g.size(),
        }
    )
    stars["n_excluded"] = stars["n_kois"] - stars["n_pos"] - stars["n_neg"]
    stars["star_stratum"] = np.select(
        [
            (stars.n_pos > 0) & (stars.n_neg == 0),
            (stars.n_neg > 0) & (stars.n_pos == 0),
            (stars.n_pos > 0) & (stars.n_neg > 0),
        ],
        ["pos_only", "neg_only", "mixed"],
        default="unlabelled_only",
    )
    return stars.reset_index()


def _cut_sizes(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of ``n`` items to splits."""
    raw = {k: n * f for k, f in fractions.items()}
    sizes = {k: math.floor(v) for k, v in raw.items()}
    remainder = n - sum(sizes.values())
    order = list(fractions)  # ties go to the earlier key
    for k in sorted(raw, key=lambda k: (raw[k] - sizes[k], -order.index(k)), reverse=True)[
        :remainder
    ]:
        sizes[k] += 1
    return sizes


def assign_star_splits(labelled: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Deterministic star → split assignment (columns: kepid, split, star_stratum, counts)."""
    stars = star_strata(labelled)
    parts = []
    for idx, stratum in enumerate(STRATA):
        group = stars[stars.star_stratum == stratum].sort_values("kepid")
        if group.empty:
            continue
        order = np.random.default_rng([cfg.seed, idx]).permutation(len(group))
        shuffled = group.iloc[order].copy()
        sizes = _cut_sizes(len(shuffled), cfg.fractions)
        labels = np.concatenate([[s] * sizes[s] for s in SPLITS])
        shuffled["split"] = labels
        parts.append(shuffled)
    return pd.concat(parts).sort_values("kepid").reset_index(drop=True)


def build_split_manifest(labelled: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Every KOI row with its star's split, label columns and original dispositions."""
    if cfg.label_policy != LABEL_POLICY:
        raise ValueError(
            f"split config expects policy {cfg.label_policy}, catalog has {LABEL_POLICY}"
        )
    stars = assign_star_splits(labelled, cfg)[["kepid", "split", "star_stratum"]]
    manifest = labelled.merge(stars, on="kepid", how="left", validate="many_to_one")
    if manifest["split"].isna().any():
        raise RuntimeError("some KOIs were not assigned a split")
    manifest = manifest[list(MANIFEST_COLUMNS)]
    return manifest.sort_values(["split", "kepid", "kepoi_name"]).reset_index(drop=True)


def check_no_leakage(manifest: pd.DataFrame) -> dict[str, int]:
    """Count shared KIC IDs between each pair of splits and duplicate KOIs; raise if any."""
    kics = {s: set(manifest.loc[manifest.split == s, "kepid"]) for s in SPLITS}
    result = {
        "shared_kic_train_val": len(kics["train"] & kics["val"]),
        "shared_kic_train_test": len(kics["train"] & kics["test"]),
        "shared_kic_val_test": len(kics["val"] & kics["test"]),
        "duplicate_koi_ids": int(manifest["kepoi_name"].duplicated().sum()),
    }
    if any(result.values()):
        raise LeakageError(f"split leakage detected: {result}")
    return result


def split_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_eligible = int((manifest.training_status == "train_eligible").sum())
    for s in SPLITS:
        m = manifest[manifest.split == s]
        eligible = m[m.training_status == "train_eligible"]
        out[s] = {
            "n_kics": int(m.kepid.nunique()),
            "n_kois": len(m),
            "n_train_eligible_kois": len(eligible),
            "n_positive": int((eligible.label == POSITIVE).sum()),
            "n_negative": int((eligible.label == NEGATIVE).sum()),
            "n_excluded": int((m.training_status == "excluded").sum()),
            "positive_fraction": round(float((eligible.label == POSITIVE).mean()), 4)
            if len(eligible)
            else None,
            "fraction_of_eligible_kois": round(len(eligible) / n_eligible, 4)
            if n_eligible
            else None,
            "stars_by_stratum": {
                k: int(v) for k, v in m.drop_duplicates("kepid").star_stratum.value_counts().items()
            },
        }
    return out


@dataclass(frozen=True)
class SplitSet:
    manifest: pd.DataFrame
    metadata: dict[str, Any]

    def kics(self, split: str) -> set[int]:
        return set(self.manifest.loc[self.manifest.split == split, "kepid"].astype(int))

    def split_of(self) -> dict[int, str]:
        return dict(zip(self.manifest.kepid.astype(int), self.manifest.split, strict=False))


def write_splits(
    manifest: pd.DataFrame,
    out_dir: Path,
    *,
    cfg: SplitConfig,
    catalog_meta: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write ``train/val/test.csv`` + metadata. Refuses to replace existing files unless asked."""
    targets = [out_dir / f"{s}.csv" for s in SPLITS] + [out_dir / METADATA_FILE]
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise SplitsExistError(
            f"split files already exist ({', '.join(p.name for p in existing)}); pass overwrite=True (CLI: --overwrite)"
        )
    leakage = check_no_leakage(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in SPLITS:
        path = out_dir / f"{s}.csv"
        manifest[manifest.split == s].to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    metadata = {
        "name": cfg.name,
        "algorithm": ALGORITHM,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "target_fractions": cfg.fractions,
        "group_key": "kepid",
        "stratification": "star stratum from train-eligible KOI labels: " + ", ".join(STRATA),
        "label_policy": cfg.label_policy,
        "source_catalog": {
            k: catalog_meta.get(k)
            for k in ("table", "query", "retrieved_at_utc", "sha256", "n_rows")
        },
        "n_kics": int(manifest.kepid.nunique()),
        "n_kois": len(manifest),
        "splits": split_summary(manifest),
        "leakage_checks": leakage,
        "files": files,
    }
    write_json_atomic(out_dir / METADATA_FILE, metadata)
    return metadata


def load_splits(out_dir: Path, *, verify: bool = True) -> SplitSet:
    """Load persisted splits, verifying file checksums and the no-leakage invariant."""
    meta_path = out_dir / METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found; run scripts/create_splits.py")
    metadata = read_json(meta_path)
    frames = []
    for s in SPLITS:
        path = out_dir / f"{s}.csv"
        if verify and sha256_file(path) != metadata["files"][path.name]:
            raise ValueError(f"{path} does not match the checksum in {meta_path}")
        frames.append(pd.read_csv(path, dtype={"label": "Int8"}, keep_default_na=True))
    manifest = pd.concat(frames, ignore_index=True)
    manifest["label_name"] = manifest["label_name"].fillna("")
    manifest["exclusion_reason"] = manifest["exclusion_reason"].fillna("")
    if verify:
        check_no_leakage(manifest)
    return SplitSet(manifest, metadata)
