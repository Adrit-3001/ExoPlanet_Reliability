"""MAST retrieval logic with injected (fake) search/download functions — no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exoreliability.config import LightCurveConfig
from exoreliability.data.cache import product_manifest_path, raw_product_path
from exoreliability.data.mast import ProductRecord, fetch_target, select_products
from tests.conftest import write_fake_kepler_fits


def _record(kepid: int, quarter: int) -> ProductRecord:
    return ProductRecord(
        kepid=kepid,
        quarter=quarter,
        filename=f"kplr{kepid:09d}-q{quarter:02d}_llc.fits",
        data_uri=f"mast:KEPLER/fake/{quarter}",
        size_bytes=1000,
        obs_id=f"kplr{kepid:09d}_lc",
        author="Kepler",
        exptime_seconds=1800.0,
    )


class FakeMast:
    """Records calls; 'downloads' write a small synthetic FITS file."""

    def __init__(self, quarters=(1, 2, 3), fail_on=None):
        self.quarters = quarters
        self.fail_on = fail_on
        self.searches = 0
        self.downloads: list[str] = []

    def search(self, kepid, cfg):
        self.searches += 1
        return [_record(kepid, q) for q in self.quarters]

    def download(self, uri: str, dest: Path) -> None:
        if self.fail_on and uri.endswith(f"/{self.fail_on}"):
            raise OSError("simulated network failure")
        self.downloads.append(uri)
        t = np.linspace(130, 131, 20)
        write_fake_kepler_fits(
            dest, kepid=1, quarter=int(uri.rsplit("/", 1)[1]), time=t, flux=np.ones(20)
        )


def test_fetch_downloads_then_uses_cache_without_network(tmp_paths):
    mast = FakeMast()
    cfg = LightCurveConfig()
    first = fetch_target(7, cfg, tmp_paths, searcher=mast.search, downloader=mast.download)
    assert first.status == "downloaded" and len(first.products) == 3
    assert all(p.sha256 and p.local_path for p in first.products)
    assert product_manifest_path(tmp_paths, 7).exists()

    second = fetch_target(7, cfg, tmp_paths, searcher=mast.search, downloader=mast.download)
    assert second.status == "cached"
    assert mast.searches == 1 and len(mast.downloads) == 3


def test_policy_change_triggers_new_search_but_reuses_files(tmp_paths):
    mast = FakeMast()
    fetch_target(7, LightCurveConfig(), tmp_paths, searcher=mast.search, downloader=mast.download)
    result = fetch_target(
        7,
        LightCurveConfig(quarters=[1, 2]),
        tmp_paths,
        searcher=mast.search,
        downloader=mast.download,
    )
    assert result.status == "downloaded" and [p.quarter for p in result.products] == [1, 2]
    assert mast.searches == 2 and len(mast.downloads) == 3  # no re-download


def test_no_products_is_reported_not_raised(tmp_paths):
    mast = FakeMast(quarters=())
    result = fetch_target(
        7, LightCurveConfig(), tmp_paths, searcher=mast.search, downloader=mast.download
    )
    assert result.status == "no_products" and not result.ok


def test_search_exception_is_reported(tmp_paths):
    def boom(kepid, cfg):
        raise ConnectionError("offline")

    result = fetch_target(
        7, LightCurveConfig(), tmp_paths, searcher=boom, downloader=FakeMast().download
    )
    assert result.status == "error" and "offline" in result.message


def test_download_failure_leaves_no_manifest(tmp_paths):
    mast = FakeMast(fail_on=2)
    result = fetch_target(
        7, LightCurveConfig(), tmp_paths, searcher=mast.search, downloader=mast.download
    )
    assert result.status == "error"
    assert not product_manifest_path(tmp_paths, 7).exists()
    assert not raw_product_path(tmp_paths, 7, _record(7, 2).filename).exists()


def test_budget_prevents_download(tmp_paths):
    mast = FakeMast()
    result = fetch_target(
        7,
        LightCurveConfig(),
        tmp_paths,
        searcher=mast.search,
        downloader=mast.download,
        remaining_budget_bytes=500,
    )
    assert result.status == "error" and mast.downloads == []


def test_select_products_filters_quarters_and_caps():
    recs = [_record(1, q) for q in (5, 1, 3, 3)]
    assert [r.quarter for r in select_products(recs, LightCurveConfig())] == [1, 3, 5]
    assert [r.quarter for r in select_products(recs, LightCurveConfig(quarters=[3, 5]))] == [3, 5]
    assert [
        r.quarter for r in select_products(recs, LightCurveConfig(max_products_per_target=2))
    ] == [1, 3]
