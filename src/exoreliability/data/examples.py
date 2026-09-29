"""KOI-level example construction and on-disk dataset storage.

Learning unit: **one KOI signal = one example**. A star's light curve is preprocessed once
(every catalogued KOI on the star masked during the trend fit), then folded separately on
each KOI's own catalog ephemeris. Examples keep their star's KIC ID and pre-assigned
split, so all examples from one star live in the same split.

Dataset directory (``data/processed/datasets/<name>/``):

* ``examples.parquet`` — one row per example: identifiers, split, label + original
  dispositions, catalog/stellar metadata, coverage and contamination diagnostics,
  ``example_status`` and source files.
* ``global_flux.npy`` / ``global_count.npy`` — float32 / int32 ``[N, global_bins]``
  (median of ``flux - 1`` per bin; NaN where the bin is empty).
* ``local_flux.npy`` / ``local_count.npy`` — the same for the local view.
* ``lightcurves/kic_*.parquet`` + ``.json`` — the detrended star light curves and their
  preprocessing records (inputs for later perturbation experiments).
* ``build_log.csv`` — one row per star with build status.
* ``dataset.json`` — config, provenance (catalog + split checksums), counts, file checksums.

``.npy`` was chosen because fixed-length float arrays for thousands of examples are a few
tens of MB, ``numpy.load(mmap_mode="r")`` avoids loading everything into memory, and no
new dependency is needed. Metadata stays in Parquet for typed columns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from exoreliability import __version__
from exoreliability.config import DatasetBuildConfig
from exoreliability.data.cache import kic_dirname, read_json, sha256_file, write_json_atomic
from exoreliability.data.catalog import ephemeris_from_row
from exoreliability.data.labels import LABEL_POLICY
from exoreliability.preprocessing.phase_fold import in_transit_mask
from exoreliability.preprocessing.pipeline import preprocess_files
from exoreliability.preprocessing.views import (
    global_view,
    local_view,
    other_koi_contamination,
    transit_coverage,
)

logger = logging.getLogger(__name__)

META_FROM_CATALOG = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "label",
    "label_name",
    "training_status",
    "exclusion_reason",
    "label_policy",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_time0bk",
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
    "koi_count",
)

STATUS_OK = "ok"


def n_transits_observed(time: np.ndarray, eph: Any) -> int:
    """Number of distinct transit epochs with at least one in-transit sample."""
    inside = in_transit_mask(time, eph, duration_factor=1.0)
    if not inside.any():
        return 0
    epochs = np.round((time[inside] - eph.epoch_bkjd) / eph.period_days)
    return int(np.unique(epochs).size)


ARRAY_FILES = ("global_flux.npy", "global_count.npy", "local_flux.npy", "local_count.npy")


@dataclass
class StarResult:
    kepid: int
    status: str  # "ok", "no_lightcurve", "preprocessing_failed"
    message: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )


def _nan_arrays(cfg: DatasetBuildConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = cfg.representation
    return (
        np.full(r.global_bins, np.nan, np.float32),
        np.zeros(r.global_bins, np.int32),
        np.full(r.local_bins, np.nan, np.float32),
        np.zeros(r.local_bins, np.int32),
    )


def build_star(
    kepid: int,
    kois: pd.DataFrame,
    split: str,
    files: list[Path],
    cfg: DatasetBuildConfig,
    *,
    flux_column: str,
    lightcurve_dir: Path | None = None,
) -> StarResult:
    """Preprocess one star and fold it on every KOI. Never raises for per-star problems."""
    if not files:
        return StarResult(kepid, "no_lightcurve", "no downloaded light-curve products")
    ephs = {r.kepoi_name: ephemeris_from_row(r) for _, r in kois.iterrows()}
    known = [e for e in ephs.values() if e is not None]
    try:
        processed = preprocess_files(
            files, cfg.preprocessing, flux_column=flux_column, ephemerides=known
        )
    except (OSError, ValueError) as exc:
        return StarResult(kepid, "preprocessing_failed", str(exc))

    frame = processed.frame
    time = frame["time_bkjd"].to_numpy(np.float64)
    flux = frame["flux_detrended"].to_numpy(np.float64)
    if lightcurve_dir is not None:
        lightcurve_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(lightcurve_dir / f"{kic_dirname(kepid)}.parquet", index=False)
        write_json_atomic(
            lightcurve_dir / f"{kic_dirname(kepid)}.json", {"kepid": kepid, **processed.record}
        )

    rep = cfg.representation
    # Stellar/instrumental variability amplitude removed by detrending (stratification aid:
    # masked detrending degrades for strongly variable stars with few transits).
    trend_rms_ppm = float(np.std(frame["trend"].to_numpy()) * 1e6)
    mask_factor = cfg.preprocessing.detrend.mask_duration_factor
    result = StarResult(kepid, "ok")
    for _, row in kois.iterrows():
        eph = ephs[row.kepoi_name]
        meta: dict[str, Any] = {c: row.get(c) for c in META_FROM_CATALOG}
        meta.update(
            split=split,
            n_kois_on_star=len(kois),
            n_points=int(time.size),
            n_quarters=int(frame["quarter"].nunique()),
            source_files=json.dumps([p.name for p in sorted(files)]),
            star_mask_fraction=float(frame["in_known_transit_mask"].mean()),
            star_trend_rms_ppm=trend_rms_ppm,
        )
        arrays = _nan_arrays(cfg)
        if eph is None or eph.duration_hours is None:
            meta["example_status"] = "missing_ephemeris"
        elif time.size < rep.min_valid_points:
            meta["example_status"] = "insufficient_points"
        else:
            g = global_view(time, flux, eph, rep.global_bins)
            loc = local_view(
                time,
                flux,
                eph,
                rep.local_bins,
                rep.local_half_width_durations,
                rep.local_bin_width_durations,
            )
            depths = dict(zip(kois.kepoi_name, kois.koi_depth, strict=True))
            others = [
                (e, depths[name])
                for name, e in ephs.items()
                if name != row.kepoi_name and e is not None
            ]
            meta.update(
                global_coverage=g.coverage,
                transit_coverage=transit_coverage(g, eph),
                local_coverage=loc.coverage,
                bins_across_transit=rep.global_bins * eph.duration_hours / 24.0 / eph.period_days,
                n_transits_observed=n_transits_observed(time, eph),
                mask_window_days=mask_factor * eph.duration_hours / 24.0,
                **other_koi_contamination(time, eph, row.koi_depth, others),
            )
            if meta["mask_window_days"] > rep.max_mask_window_days:
                # Outside the validated regime of masked detrending (docs/data_contract.md §6).
                meta["example_status"] = "detrend_mask_too_long"
            elif g.coverage < rep.min_global_coverage:
                meta["example_status"] = "insufficient_global_coverage"
            elif meta["transit_coverage"] < rep.min_transit_coverage:
                meta["example_status"] = "insufficient_transit_coverage"
            else:
                meta["example_status"] = STATUS_OK
            arrays = (g.flux, g.count, loc.flux, loc.count)
        result.rows.append(meta)
        result.arrays.append(arrays)
    return result


@dataclass(frozen=True)
class PreparedDataset:
    """A built dataset. Arrays are memory-mapped (read-only) by default."""

    root: Path
    examples: pd.DataFrame
    global_flux: np.ndarray
    global_count: np.ndarray
    local_flux: np.ndarray
    local_count: np.ndarray
    info: dict[str, Any]

    def __len__(self) -> int:
        return len(self.examples)


def write_dataset(
    out_dir: Path,
    results: list[StarResult],
    *,
    cfg: DatasetBuildConfig,
    info: dict[str, Any],
) -> dict[str, Any]:
    rows = [r for s in results for r in s.rows]
    arrays = [a for s in results for a in s.arrays]
    if not rows:
        raise ValueError("no examples to write")
    examples = pd.DataFrame(rows)
    examples.insert(0, "example_index", np.arange(len(examples)))
    examples["label"] = examples["label"].astype("Int8")
    examples.to_parquet(out_dir / "examples.parquet", index=False)
    stacks = [np.stack([a[i] for a in arrays]) for i in range(4)]
    for name, arr in zip(ARRAY_FILES, stacks, strict=True):
        np.save(out_dir / name, arr)
    log = pd.DataFrame(
        [
            {"kepid": s.kepid, "status": s.status, "message": s.message, "n_examples": len(s.rows)}
            for s in results
        ]
    )
    log.to_csv(out_dir / "build_log.csv", index=False)

    ok = examples[examples.example_status == STATUS_OK]
    trainable = ok[ok.training_status == "train_eligible"]
    doc = {
        **info,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "exoreliability_version": __version__,
        "config": cfg.model_dump(),
        "learning_unit": "one KOI signal = one example; splits grouped by kepid",
        "label_policy": LABEL_POLICY,
        "shapes": {n: list(a.shape) for n, a in zip(ARRAY_FILES, stacks, strict=True)},
        "flux_definition": "median of (detrended flux - 1) per bin; NaN = empty bin",
        "counts": {
            "n_examples": len(examples),
            "n_examples_ok": len(ok),
            "n_trainable": len(trainable),
            "n_stars_attempted": len(results),
            "n_stars_ok": int((log.status == "ok").sum()),
            "example_status": {
                k: int(v) for k, v in examples.example_status.value_counts().items()
            },
            "star_status": {k: int(v) for k, v in log.status.value_counts().items()},
            "trainable_by_split": {
                s: {
                    "n": int((trainable.split == s).sum()),
                    "n_positive": int(((trainable.split == s) & (trainable.label == 1)).sum()),
                    "n_negative": int(((trainable.split == s) & (trainable.label == 0)).sum()),
                    "n_kics": int(trainable[trainable.split == s].kepid.nunique()),
                }
                for s in ("train", "val", "test")
            },
        },
        "files": {n: sha256_file(out_dir / n) for n in ("examples.parquet", *ARRAY_FILES)},
    }
    write_json_atomic(out_dir / "dataset.json", doc)
    return doc


def load_dataset(root: Path, *, mmap: bool = True, verify: bool = False) -> PreparedDataset:
    info = read_json(root / "dataset.json")
    if verify:
        for name, digest in info["files"].items():
            if sha256_file(root / name) != digest:
                raise ValueError(f"{root / name} does not match its checksum in dataset.json")
    mode: Literal["r"] | None = "r" if mmap else None
    gf, gc, lf, lc = (np.load(root / n, mmap_mode=mode) for n in ARRAY_FILES)
    examples = pd.read_parquet(root / "examples.parquet")
    if len(examples) != gf.shape[0]:
        raise ValueError("examples.parquet and array files disagree on the number of examples")
    return PreparedDataset(root, examples, gf, gc, lf, lc, info=info)
