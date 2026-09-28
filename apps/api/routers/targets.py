"""Target endpoints. All data is read from local files via ``TargetRepository``; no downloads."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from apps.api.dependencies import get_repository
from apps.api.schemas.targets import (
    KOI,
    BLSResponse,
    BLSRun,
    CatalogSource,
    LightCurveResponse,
    Periodogram,
    PhaseFoldResponse,
    PreprocessingSummary,
    ProductSummary,
    TargetDetail,
    TargetList,
    TargetSummary,
)
from exoreliability.data.cache import read_json
from exoreliability.data.catalog import ephemeris_from_row
from exoreliability.data.contracts import Ephemeris
from exoreliability.preprocessing.phase_fold import phase_fold
from exoreliability.preprocessing.resample import bin_curve
from exoreliability.targets import TargetRepository, parse_target_id

router = APIRouter(prefix="/targets", tags=["targets"])

TargetId = Path(..., description="KIC id, e.g. 10811496 or 'KIC 10811496'")


def _kepid(target_id: str) -> int:
    try:
        return parse_target_id(target_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _kois(repo: TargetRepository, kepid: int) -> list[KOI]:
    out = []
    for _, row in repo.kois(kepid).iterrows():
        data = {f: _clean(row.get(f)) for f in KOI.model_fields}
        # Empty strings mean "not applicable" in the derived-label columns.
        data["label_name"] = data["label_name"] or None
        data["exclusion_reason"] = data["exclusion_reason"] or None
        out.append(KOI(**data))
    return out


def _bls_run(run: dict[str, Any]) -> BLSRun:
    return BLSRun(
        perturbation=run["perturbation"],
        severity=run["severity"],
        sigma_added=run["perturbation_params"]["sigma_added"],
        stats=run["bls"],
        catalog_comparison=run["catalog_comparison"],
    )


def _summary(repo: TargetRepository, kepid: int) -> TargetSummary:
    kois = repo.kois(kepid)
    return TargetSummary(
        target_id=f"KIC {kepid}",
        kepid=kepid,
        n_kois=len(kois),
        dispositions=sorted(set(kois["koi_disposition"].dropna())) if len(kois) else [],
        has_lightcurve=repo.processed(kepid) is not None,
        has_bls=repo.latest_bls(kepid) is not None,
    )


@router.get("", response_model=TargetList)
def list_targets(repo: TargetRepository = Depends(get_repository)) -> TargetList:
    """Targets with a locally processed light curve."""
    return TargetList(targets=[_summary(repo, k) for k in repo.processed_kepids()])


@router.get("/{target_id}", response_model=TargetDetail)
def get_target(
    target_id: str = TargetId, repo: TargetRepository = Depends(get_repository)
) -> TargetDetail:
    kepid = _kepid(target_id)
    kois = _kois(repo, kepid)
    manifest = repo.product_manifest(kepid)
    processed = repo.processed(kepid)
    if not kois and manifest is None and processed is None:
        raise HTTPException(
            status_code=404, detail=f"KIC {kepid} is not in the local catalog snapshot or data"
        )

    meta = repo.catalog_meta()
    products = None
    if manifest is not None and manifest["products"]:
        first = manifest["products"][0]
        products = ProductSummary(
            n_products=len(manifest["products"]),
            quarters=[p["quarter"] for p in manifest["products"]],
            author=first["author"],
            exptime_seconds=first["exptime_seconds"],
            retrieved_at_utc=manifest.get("retrieved_at_utc"),
        )
    preprocessing = None
    if processed is not None:
        rec = processed.record
        preprocessing = PreprocessingSummary(
            created_at_utc=rec.get("created_at_utc"),
            flux_column=rec.get("flux_column", ""),
            n_points=len(processed.frame),
            data_release=sorted({str(q.get("data_release")) for q in rec.get("quarters", [])}),
            config=rec.get("config", {}),
        )
    bls = repo.latest_bls(kepid)
    clean_run = None
    if bls is not None:
        clean_run = next(
            (_bls_run(r) for r in bls[1]["runs"] if r.get("status") == "ok" and r["severity"] == 0),
            None,
        )
    summary = _summary(repo, kepid)
    return TargetDetail(
        **summary.model_dump(),
        kois=kois,
        catalog_source=(
            CatalogSource(
                table=meta["table"], retrieved_at_utc=meta["retrieved_at_utc"], query=meta["query"]
            )
            if meta
            else None
        ),
        products=products,
        preprocessing=preprocessing,
        bls_clean=clean_run,
        bls_run_id=bls[1]["run_id"] if bls else None,
    )


@router.get("/{target_id}/lightcurve", response_model=LightCurveResponse)
def get_lightcurve(
    target_id: str = TargetId,
    max_points: int | None = Query(
        None, ge=100, le=500_000, description="bin in time to at most this many points"
    ),
    repo: TargetRepository = Depends(get_repository),
) -> LightCurveResponse:
    kepid = _kepid(target_id)
    processed = repo.processed(kepid)
    if processed is None:
        raise HTTPException(status_code=404, detail=f"no processed light curve for KIC {kepid}")
    f = processed.frame
    t = f["time_bkjd"].to_numpy()
    if max_points is None or len(f) <= max_points:
        return LightCurveResponse(
            kepid=kepid,
            n_points=len(f),
            binned=False,
            bin_width_days=None,
            time=t.tolist(),
            flux_norm=f["flux_norm"].tolist(),
            trend=f["trend"].tolist(),
            flux_detrended=f["flux_detrended"].tolist(),
            quarter=f["quarter"].astype(int).tolist(),
        )
    rng = (float(t.min()), float(t.max()))
    cols = {}
    for c in ("time_bkjd", "flux_norm", "trend", "flux_detrended", "quarter"):
        _, cols[c], counts = bin_curve(
            t, f[c].to_numpy(np.float64), max_points, rng, statistic="mean"
        )
    keep = counts > 0
    return LightCurveResponse(
        kepid=kepid,
        n_points=int(keep.sum()),
        binned=True,
        bin_width_days=(rng[1] - rng[0]) / max_points,
        time=cols["time_bkjd"][keep].tolist(),
        flux_norm=cols["flux_norm"][keep].tolist(),
        trend=cols["trend"][keep].tolist(),
        flux_detrended=cols["flux_detrended"][keep].tolist(),
        quarter=np.rint(cols["quarter"][keep]).astype(int).tolist(),
    )


@router.get("/{target_id}/phase-folded", response_model=PhaseFoldResponse)
def get_phase_folded(
    target_id: str = TargetId,
    koi: str | None = Query(
        None, description="KOI name for catalog ephemeris (default: first KOI)"
    ),
    source: Literal["catalog", "bls"] = Query("catalog"),
    window_hours: float = Query(24.0, gt=0, le=240),
    bins: int = Query(120, ge=10, le=2000),
    repo: TargetRepository = Depends(get_repository),
) -> PhaseFoldResponse:
    kepid = _kepid(target_id)
    processed = repo.processed(kepid)
    if processed is None:
        raise HTTPException(status_code=404, detail=f"no processed light curve for KIC {kepid}")
    kepoi_name: str | None = None
    if source == "catalog":
        kois = repo.kois(kepid)
        if koi is not None:
            kois = kois[kois["kepoi_name"] == koi]
        if kois.empty:
            raise HTTPException(
                status_code=404, detail=f"no catalog KOI {koi or ''} for KIC {kepid}".strip()
            )
        row = kois.iloc[0]
        eph = ephemeris_from_row(row)
        if eph is None:
            raise HTTPException(status_code=404, detail="catalog ephemeris unavailable")
        kepoi_name = str(row["kepoi_name"])
    else:
        bls = repo.latest_bls(kepid)
        clean = (
            next(
                (r for r in bls[1]["runs"] if r.get("severity") == 0 and r.get("status") == "ok"),
                None,
            )
            if bls
            else None
        )
        if clean is None:
            raise HTTPException(status_code=404, detail=f"no BLS result for KIC {kepid}")
        s = clean["bls"]
        eph = Ephemeris(s["period_days"], s["transit_time_bkjd"], s["duration_hours"])

    t = processed.frame["time_bkjd"].to_numpy()
    y = processed.frame["flux_detrended"].to_numpy()
    hours = phase_fold(t, eph) * 24.0
    window = min(window_hours, 12.0 * eph.period_days)
    sel = np.abs(hours) <= window
    order = np.argsort(hours[sel])
    centres, med, _ = bin_curve(hours[sel], y[sel], bins, (-window, window))
    return PhaseFoldResponse(
        kepid=kepid,
        kepoi_name=kepoi_name,
        ephemeris_source=source,
        period_days=eph.period_days,
        epoch_bkjd=eph.epoch_bkjd,
        duration_hours=eph.duration_hours,
        window_hours=window,
        phase_hours=hours[sel][order].tolist(),
        flux=y[sel][order].tolist(),
        binned_phase_hours=centres.tolist(),
        binned_flux=[None if np.isnan(v) else float(v) for v in med],
    )


@router.get("/{target_id}/bls", response_model=BLSResponse)
def get_bls(
    target_id: str = TargetId, repo: TargetRepository = Depends(get_repository)
) -> BLSResponse:
    kepid = _kepid(target_id)
    found = repo.latest_bls(kepid)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no BLS run found for KIC {kepid}")
    run_dir, entry = found
    runs = [r for r in entry["runs"] if r.get("status") == "ok"]
    clean = next((r for r in runs if r["severity"] == 0), None)
    periodogram = None
    if clean is not None and (run_dir / clean["periodogram_file"]).exists():
        periodogram = Periodogram(**read_json(run_dir / clean["periodogram_file"]))
    return BLSResponse(
        kepid=kepid,
        run_id=entry["run_id"],
        clean=_bls_run(clean) if clean else None,
        noise_demo=[_bls_run(r) for r in runs if r["severity"] > 0],
        periodogram=periodogram,
        note="Single-target pipeline validation; not a statistical performance estimate.",
    )
