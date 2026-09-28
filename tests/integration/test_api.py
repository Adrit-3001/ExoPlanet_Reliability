"""API tests against a SYNTHETIC fixture project on disk (no network, no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import get_repository
from apps.api.main import app
from exoreliability.baselines.bls import compare_periods, downsample_periodogram, run_bls
from exoreliability.config import BLSConfig, PreprocessingConfig
from exoreliability.data.archive import DR25_KOI_COLUMNS, DR25_KOI_TABLE
from exoreliability.data.cache import catalog_meta_path, catalog_snapshot_path, write_json_atomic
from exoreliability.preprocessing.pipeline import preprocess_files, save_processed
from exoreliability.targets import TargetRepository, parse_target_id
from tests.conftest import make_transit_lc, write_fake_kepler_fits

KEPID = 1234567
PERIOD, EPOCH = 3.7, 131.3


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from exoreliability.config import ProjectPaths

    paths = ProjectPaths(root=tmp_path_factory.mktemp("project"))

    # Catalog snapshot (synthetic rows).
    rows = []
    for name, disp, pdisp, period in [
        ("K09999.01", "CONFIRMED", "CANDIDATE", PERIOD),
        ("K09999.02", "CANDIDATE", "CANDIDATE", 11.0),
    ]:
        row = dict.fromkeys(DR25_KOI_COLUMNS)
        row.update(
            kepid=KEPID,
            kepoi_name=name,
            koi_disposition=disp,
            koi_pdisposition=pdisp,
            koi_period=period,
            koi_time0bk=EPOCH,
            koi_duration=3.0,
            koi_depth=1000.0,
        )
        rows.append(row)
    snap = catalog_snapshot_path(paths, DR25_KOI_TABLE)
    snap.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(snap, index=False)
    write_json_atomic(
        catalog_meta_path(paths, DR25_KOI_TABLE),
        {
            "table": DR25_KOI_TABLE,
            "retrieved_at_utc": "2000-01-01T00:00:00+00:00",
            "query": "select",
        },
    )

    # Processed light curve from a synthetic FITS file.
    lc = make_transit_lc(n_days=40, period=PERIOD, epoch=EPOCH - 130.0, depth=1e-3, noise=2e-4)
    fits_path = write_fake_kepler_fits(
        paths.root / "raw.fits",
        kepid=KEPID,
        quarter=1,
        time=lc.time,
        flux=lc.flux * 1e4,
        flux_err=lc.flux_err * 1e4,
    )
    processed = preprocess_files([fits_path], PreprocessingConfig())
    save_processed(processed, paths, KEPID)

    # A BLS run folder in the same format written by scripts/run_bls.py.
    cfg = BLSConfig(
        min_period_days=1.0,
        max_period_days=10.0,
        durations_hours=[2.0, 3.0],
        oversample=2,
        n_jobs=1,
    )
    result, pg = run_bls(processed.to_lightcurve(), cfg)
    run_dir = paths.experiments / "2000-01-01T000000Z_bls_test"
    write_json_atomic(
        run_dir / "periodograms" / "p.json", downsample_periodogram(pg["period"], pg["power"])
    )
    run = {
        "status": "ok",
        "perturbation": "gaussian_noise",
        "severity": 0.0,
        "perturbation_params": {"sigma_added": 0.0},
        "bls": result.to_dict(),
        "catalog_comparison": [
            {
                "kepoi_name": "K09999.01",
                "catalog_period": PERIOD,
                **compare_periods(result.period_days, PERIOD),
            }
        ],
        "periodogram_file": "periodograms/p.json",
    }
    write_json_atomic(
        run_dir / "results.json",
        {
            "kind": "bls",
            "run_id": run_dir.name,
            "targets": [{"kepid": KEPID, "status": "ok", "kois": [], "runs": [run]}],
        },
    )

    app.dependency_overrides[get_repository] = lambda: TargetRepository(paths)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert (
        body["status"] == "ok"
        and body["catalog_snapshot_available"]
        and body["processed_targets"] == 1
    )


def test_list_targets(client):
    targets = client.get("/targets").json()["targets"]
    assert [t["kepid"] for t in targets] == [KEPID]
    assert targets[0]["has_lightcurve"] and targets[0]["has_bls"]


@pytest.mark.parametrize("tid", [str(KEPID), f"KIC {KEPID}", f"kic_{KEPID:09d}"])
def test_get_target_accepts_id_forms(client, tid):
    r = client.get(f"/targets/{tid}")
    assert r.status_code == 200
    body = r.json()
    assert body["kepid"] == KEPID and body["n_kois"] == 2
    k1, k2 = body["kois"]
    assert k1["label_name"] == "planet" and k2["label_name"] is None
    assert k2["exclusion_reason"] == "unresolved_candidate"
    assert body["bls_clean"]["stats"]["period_days"] == pytest.approx(PERIOD, rel=2e-3)
    assert body["bls_clean"]["catalog_comparison"][0]["relation"] == "match"


def test_unknown_target_404(client):
    assert client.get("/targets/999").status_code == 404
    assert client.get("/targets/999/lightcurve").status_code == 404
    assert client.get("/targets/999/bls").status_code == 404


@pytest.mark.parametrize("bad", ["abc", "KIC-12x", "1234567890123"])
def test_invalid_target_id_422(client, bad):
    assert client.get(f"/targets/{bad}").status_code == 422


def test_lightcurve_full_and_binned(client):
    full = client.get(f"/targets/{KEPID}/lightcurve").json()
    assert not full["binned"] and full["n_points"] == len(full["time"]) == len(
        full["flux_detrended"]
    )
    assert np.all(np.diff(full["time"]) > 0)
    binned = client.get(f"/targets/{KEPID}/lightcurve", params={"max_points": 200}).json()
    assert binned["binned"] and 0 < binned["n_points"] <= 200


def test_lightcurve_invalid_query_422(client):
    assert client.get(f"/targets/{KEPID}/lightcurve", params={"max_points": 5}).status_code == 422
    assert client.get(f"/targets/{KEPID}/lightcurve", params={"max_points": "x"}).status_code == 422


def test_phase_folded_catalog_and_bls(client):
    r = client.get(
        f"/targets/{KEPID}/phase-folded",
        params={"koi": "K09999.01", "window_hours": 12, "bins": 48},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ephemeris_source"] == "catalog" and body["period_days"] == PERIOD
    assert len(body["binned_flux"]) == 48
    assert max(abs(h) for h in body["phase_hours"]) <= 12
    # The binned curve dips at phase 0 (synthetic 1000 ppm transit).
    centre = min(range(48), key=lambda i: abs(body["binned_phase_hours"][i]))
    assert body["binned_flux"][centre] < 0.9995

    bls = client.get(f"/targets/{KEPID}/phase-folded", params={"source": "bls"}).json()
    assert bls["ephemeris_source"] == "bls"
    assert (
        client.get(f"/targets/{KEPID}/phase-folded", params={"koi": "K00000.01"}).status_code == 404
    )
    assert (
        client.get(f"/targets/{KEPID}/phase-folded", params={"source": "magic"}).status_code == 422
    )


def test_bls_endpoint(client):
    body = client.get(f"/targets/{KEPID}/bls").json()
    assert body["clean"]["severity"] == 0.0
    assert body["noise_demo"] == []
    assert len(body["periodogram"]["period"]) == len(body["periodogram"]["power"])


def test_parse_target_id():
    assert parse_target_id("KIC 10811496") == 10811496
    assert parse_target_id("kic_010811496") == 10811496
    with pytest.raises(ValueError):
        parse_target_id("TIC 123")
