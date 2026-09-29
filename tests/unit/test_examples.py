"""KOI-level views, multi-KOI examples, missing data, storage and PyTorch loading.

All light curves are SYNTHETIC (box transits + white noise written as fake Kepler FITS).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from exoreliability.config import (
    DatasetBuildConfig,
    DetrendConfig,
    PreprocessingConfig,
    RepresentationConfig,
)
from exoreliability.data.contracts import Ephemeris
from exoreliability.data.examples import STATUS_OK, build_star, load_dataset, write_dataset
from exoreliability.data.labels import apply_label_policy
from exoreliability.preprocessing.views import (
    global_view,
    local_view,
    other_koi_contamination,
    transit_coverage,
    windowed_median,
)
from tests.conftest import KEPLER_LC_CADENCE_DAYS, write_fake_kepler_fits

T0 = 130.0
KOI_A = Ephemeris(3.7, T0 + 1.3, 3.0)
KOI_B = Ephemeris(9.1, T0 + 2.9, 4.0)


def box(t: np.ndarray, eph: Ephemeris, depth: float) -> np.ndarray:
    ph = np.mod(t - eph.epoch_bkjd + 0.5 * eph.period_days, eph.period_days) - 0.5 * eph.period_days
    return np.where(np.abs(ph) < 0.5 * eph.duration_hours / 24, depth, 0.0)


def synthetic_star(tmp_path, *, n_days=180.0, gap=None, seed=0, kepid=42):
    """Two quarters of a star hosting KOI_A (1000 ppm) and KOI_B (2500 ppm). SYNTHETIC."""
    rng = np.random.default_rng(seed)
    files = []
    for q, (start, stop) in enumerate(((0, n_days / 2), (n_days / 2 + 2, n_days)), start=1):
        t = T0 + np.arange(start, stop, KEPLER_LC_CADENCE_DAYS)
        if gap is not None:
            t = t[(t < gap[0]) | (t > gap[1])]
        flux = 1.0 - box(t, KOI_A, 1e-3) - box(t, KOI_B, 2.5e-3) + rng.normal(0, 1e-4, t.size)
        files.append(
            write_fake_kepler_fits(
                tmp_path / f"q{q}.fits",
                kepid=kepid,
                quarter=q,
                time=t,
                flux=flux * 1e4,
                flux_err=np.full(t.size, 1.0),
            )
        )
    return files


def star_kois(kepid=42, dispositions=(("CONFIRMED", "CANDIDATE"), ("CANDIDATE", "CANDIDATE"))):
    rows = []
    for name, eph, depth, disp in (
        ("K00042.01", KOI_A, 1000.0, dispositions[0]),
        ("K00042.02", KOI_B, 2500.0, dispositions[1]),
    ):
        rows.append(
            {
                "kepid": kepid,
                "kepoi_name": name,
                "koi_disposition": disp[0],
                "koi_pdisposition": disp[1],
                "koi_period": eph.period_days,
                "koi_time0bk": eph.epoch_bkjd,
                "koi_duration": eph.duration_hours,
                "koi_depth": depth,
                "koi_model_snr": 50.0,
                "koi_kepmag": 14.0,
            }
        )
    return apply_label_policy(pd.DataFrame(rows))


def cfg(**rep) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        scale="smoke",
        preprocessing=PreprocessingConfig(
            detrend=DetrendConfig(method="robust_spline", mask_known_transits=True)
        ),
        representation=RepresentationConfig(
            **{"global_bins": 512, "local_bins": 61, "min_valid_points": 500, **rep}
        ),
    )


# -- views -----------------------------------------------------------------------------


def test_windowed_median_counts_and_empty_bins():
    x = np.array([0.05, 0.1, 0.15, 0.6])
    y = np.array([1.0, 2.0, 3.0, 9.0])
    med, cnt = windowed_median(x, y, np.array([0.1, 0.35, 0.6]), 0.2)
    assert cnt.tolist() == [3, 0, 1]
    assert med[0] == 2.0 and np.isnan(med[1]) and med[2] == 9.0


def test_global_and_local_views_centre_the_transit():
    t = T0 + np.arange(0, 90, KEPLER_LC_CADENCE_DAYS)
    f = 1.0 - box(t, KOI_A, 1e-3)
    g = global_view(t, f, KOI_A, 256)
    loc = local_view(t, f, KOI_A, 41, 2.0, 0.16)
    assert g.flux.shape == (256,) and loc.flux.shape == (41,)
    assert g.count.sum() == t.size  # non-overlapping global bins partition the samples
    assert g.flux[128] == pytest.approx(-1e-3) and g.flux[0] == pytest.approx(0.0)
    assert loc.flux[20] == pytest.approx(-1e-3) and loc.flux[0] == pytest.approx(0.0)
    assert g.coverage == 1.0 and transit_coverage(g, KOI_A) == 1.0


def test_contamination_same_period_is_coherent_and_different_period_is_not():
    t = T0 + np.arange(0, 180, KEPLER_LC_CADENCE_DAYS)
    secondary = Ephemeris(KOI_A.period_days, KOI_A.epoch_bkjd + KOI_A.period_days / 2, 3.0)
    coherent = other_koi_contamination(t, KOI_A, 1000.0, [(secondary, 1000.0)])
    incoherent = other_koi_contamination(t, KOI_A, 1000.0, [(KOI_B, 1000.0)])
    assert coherent["contam_max_bin_fraction"] > 0.9
    assert coherent["contam_max_relative_to_depth"] > 0.9
    assert incoherent["contam_max_relative_to_depth"] < 0.35
    assert other_koi_contamination(t, KOI_A, 1000.0, [])["contam_max_bin_fraction"] == 0.0


# -- multi-KOI stars -----------------------------------------------------------------------


def test_one_star_yields_one_example_per_koi_with_own_ephemeris(tmp_path):
    res = build_star(
        42, star_kois(), "train", synthetic_star(tmp_path), cfg(), flux_column="PDCSAP_FLUX"
    )
    assert res.status == "ok" and len(res.rows) == 2
    assert {r["kepoi_name"] for r in res.rows} == {"K00042.01", "K00042.02"}
    assert all(
        r["kepid"] == 42 and r["split"] == "train" and r["n_kois_on_star"] == 2 for r in res.rows
    )
    for row, (g, _gc, _l, _lc), depth in zip(res.rows, res.arrays, (1e-3, 2.5e-3), strict=True):
        assert row["example_status"] == STATUS_OK
        centre = g.size // 2
        # Each view is folded on its own KOI: its own dip sits at phase 0 ...
        assert g[centre] == pytest.approx(-depth, rel=0.1)
        # ... and the other KOI's transits do not produce a comparable coherent dip elsewhere
        # (±12 bins excludes this KOI's own ingress/egress: 3 h = ±8.6 bins at 512 bins).
        assert np.nanmin(np.delete(g, np.arange(centre - 12, centre + 13))) > -0.5 * depth


def test_candidate_sibling_kept_but_flagged(tmp_path):
    res = build_star(
        42, star_kois(), "val", synthetic_star(tmp_path), cfg(), flux_column="PDCSAP_FLUX"
    )
    by = {r["kepoi_name"]: r for r in res.rows}
    assert by["K00042.01"]["training_status"] == "train_eligible" and by["K00042.01"]["label"] == 1
    assert by["K00042.02"]["training_status"] == "excluded"
    assert by["K00042.02"]["exclusion_reason"] == "unresolved_candidate"
    assert by["K00042.02"]["koi_disposition"] == "CANDIDATE"  # original disposition preserved


# -- missing data ------------------------------------------------------------------------


def test_large_phase_gap_is_flagged(tmp_path):
    # Only ~1.2 d of data: a 9.1-d period is covered over far less than half its phase.
    files = [
        write_fake_kepler_fits(
            tmp_path / "q1.fits",
            kepid=42,
            quarter=1,
            time=T0 + 2.5 + np.arange(0, 1.2, KEPLER_LC_CADENCE_DAYS),
            flux=np.full(59, 1e4),
        )
    ]
    res = build_star(42, star_kois(), "train", files, cfg(), flux_column="PDCSAP_FLUX")
    statuses = {r["kepoi_name"]: r["example_status"] for r in res.rows}
    assert statuses["K00042.02"] == "insufficient_points" or statuses["K00042.02"].startswith(
        "insufficient"
    )


def test_insufficient_global_coverage(tmp_path):
    files = synthetic_star(tmp_path, n_days=6.0)  # < one period of KOI_B
    res = build_star(
        42, star_kois(), "train", files, cfg(min_valid_points=10), flux_column="PDCSAP_FLUX"
    )
    b = next(r for r in res.rows if r["kepoi_name"] == "K00042.02")
    assert b["global_coverage"] < 0.5
    assert b["example_status"] == "insufficient_global_coverage"


def test_no_lightcurve_and_missing_ephemeris(tmp_path):
    assert (
        build_star(42, star_kois(), "train", [], cfg(), flux_column="PDCSAP_FLUX").status
        == "no_lightcurve"
    )
    kois = star_kois()
    kois.loc[1, "koi_duration"] = np.nan
    res = build_star(42, kois, "train", synthetic_star(tmp_path), cfg(), flux_column="PDCSAP_FLUX")
    assert {r["kepoi_name"]: r["example_status"] for r in res.rows}[
        "K00042.02"
    ] == "missing_ephemeris"


# -- storage + PyTorch --------------------------------------------------------------------


@pytest.fixture
def built(tmp_path):
    results = []
    for i, (kepid, split) in enumerate(((42, "train"), (43, "train"), (44, "val"), (45, "test"))):
        d = tmp_path / f"star{kepid}"
        d.mkdir()
        disp = (
            (("CONFIRMED", "CANDIDATE"), ("FALSE POSITIVE", "FALSE POSITIVE"))
            if i % 2 == 0
            else (("FALSE POSITIVE", "FALSE POSITIVE"), ("CANDIDATE", "CANDIDATE"))
        )
        kois = star_kois(kepid, disp).assign(
            kepoi_name=lambda df, k=kepid: [f"K{k:05d}.01", f"K{k:05d}.02"]
        )
        results.append(
            build_star(
                kepid,
                kois,
                split,
                synthetic_star(d, seed=i, kepid=kepid),
                cfg(),
                flux_column="PDCSAP_FLUX",
            )
        )
    out = tmp_path / "ds"
    out.mkdir()
    doc = write_dataset(out, results, cfg=cfg(), info={"name": "synthetic"})
    return out, doc


def test_dataset_roundtrip_and_metadata(built):
    out, doc = built
    d = load_dataset(out, verify=True)
    assert d.global_flux.shape == (8, 512) and d.local_flux.shape == (8, 61)
    assert doc["counts"]["n_examples"] == 8
    ex = d.examples
    assert list(ex.example_index) == list(range(8))
    row = ex[ex.kepoi_name == "K00044.02"].iloc[0]
    assert row.kepid == 44 and row.split == "val" and row.koi_period == KOI_B.period_days
    assert json.loads(row.source_files) == ["q1.fits", "q2.fits"]
    # No star in more than one split.
    assert (ex.groupby("kepid").split.nunique() == 1).all()


def test_fill_missing_policies():
    from exoreliability.training.dataset import fill_missing, normalize_view

    v = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    full, mask = fill_missing(v, np.ones(4, int), circular=False)
    np.testing.assert_array_equal(full, v)
    assert mask.tolist() == [1, 1, 1, 1]
    gap = np.array([0.0, np.nan, np.nan, 3.0], np.float32)
    filled, mask = fill_missing(gap, np.array([1, 0, 0, 1]), circular=False)
    np.testing.assert_allclose(filled, [0, 1, 2, 3])
    assert mask.tolist() == [1, 0, 0, 1]
    circ, _ = fill_missing(
        np.array([np.nan, 2.0, 4.0, np.nan], np.float32), np.array([0, 1, 1, 0]), circular=True
    )
    np.testing.assert_allclose(
        circ, [8 / 3, 2.0, 4.0, 10 / 3], rtol=1e-6
    )  # wraps: bins 3, 0 interpolate 4 -> 2
    empty, emask = fill_missing(np.full(3, np.nan, np.float32), np.zeros(3, int), circular=True)
    assert empty.tolist() == [0, 0, 0] and emask.tolist() == [0, 0, 0]
    norm = normalize_view(
        np.array([0.0, 0.0, -2e-3, 0.0], np.float32), np.ones(4, np.float32), "depth"
    )
    np.testing.assert_allclose(norm, [0, 0, -1, 0])


def test_torch_dataset_and_loader(built):
    torch = pytest.importorskip("torch")
    from exoreliability.training.dataset import KOIDataset, make_dataloader

    out, _ = built
    train = KOIDataset(out, "train", views=("global", "local"), return_metadata=True)
    # Train stars 42, 43: trainable = 42.01 (planet), 42.02 (FP), 43.01 (FP); 43.02 is a candidate.
    assert sorted(train.examples.kepoi_name) == ["K00042.01", "K00042.02", "K00043.01"]
    item = train[0]
    assert item["global"].shape == (2, 512) and item["local"].shape == (2, 61)
    assert item["global"].dtype == torch.float32
    assert float(item["global"][0].min()) == pytest.approx(-1.0)
    labels = {r.kepoi_name: r.label for r in train.examples.itertuples()}
    for i in range(len(train)):
        name = train[i]["meta"]["kepoi_name"]
        assert float(train[i]["label"]) == labels[name]

    batch = next(iter(make_dataloader(train, batch_size=2, shuffle=True, seed=1)))
    assert batch["global"].shape == (2, 2, 512) and batch["label"].shape == (2,)
    again = next(iter(make_dataloader(train, batch_size=2, shuffle=True, seed=1)))
    assert batch["meta"]["kepoi_name"] == again["meta"]["kepoi_name"]  # seeded order

    everything = KOIDataset(out, None, trainable_only=False)
    assert len(everything) == 8
    assert sum(torch.isnan(everything[i]["label"]).item() for i in range(8)) == 2
    no_mask = KOIDataset(out, "val", include_mask=False, normalization="none")
    assert no_mask[0]["global"].shape == (1, 512)
    with pytest.raises(ValueError):
        KOIDataset(out, "holdout")


def test_long_mask_window_is_flagged_and_diagnostics_recorded(tmp_path):
    kois = star_kois()
    kois.loc[1, "koi_duration"] = 30.0  # 30-h transit -> 2.5-d mask window > 2 d limit
    res = build_star(42, kois, "train", synthetic_star(tmp_path), cfg(), flux_column="PDCSAP_FLUX")
    by = {r["kepoi_name"]: r for r in res.rows}
    assert by["K00042.02"]["example_status"] == "detrend_mask_too_long"
    assert by["K00042.01"]["example_status"] == STATUS_OK
    a = by["K00042.01"]
    assert a["mask_window_days"] == pytest.approx(0.25)
    assert a["n_transits_observed"] >= 40  # ~180 d / 3.7 d, minus the inter-quarter gap
    assert 0 < a["star_mask_fraction"] < 0.5
    assert a["star_trend_rms_ppm"] >= 0
