from __future__ import annotations

import httpx
import pandas as pd
import pytest

from exoreliability.config import SelectionConfig
from exoreliability.data.archive import (
    DR25_KOI_COLUMNS,
    ArchiveError,
    build_select_query,
    fetch_dr25_koi_catalog,
    run_tap_query,
)
from exoreliability.data.cache import catalog_snapshot_path, kic_dirname
from exoreliability.data.catalog import _allocate, assign_labels, ephemeris_from_row, select_subset

# SYNTHETIC catalog rows covering every disposition combination seen in DR25.
ROWS = [
    ("K1.01", 1, "CONFIRMED", "CANDIDATE"),
    ("K2.01", 2, "FALSE POSITIVE", "FALSE POSITIVE"),
    ("K3.01", 3, "CANDIDATE", "CANDIDATE"),
    ("K4.01", 4, "CONFIRMED", "FALSE POSITIVE"),
    ("K5.01", 5, "FALSE POSITIVE", "CANDIDATE"),
    ("K6.01", 6, None, "CANDIDATE"),
]


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        ROWS, columns=["kepoi_name", "kepid", "koi_disposition", "koi_pdisposition"]
    )


def test_label_mapping_dr25_conservative_v1():
    out = assign_labels(_catalog()).set_index("kepoi_name")
    assert out.loc["K1.01", "label"] == 1 and out.loc["K1.01", "label_name"] == "planet"
    assert out.loc["K2.01", "label"] == 0 and out.loc["K2.01", "label_name"] == "false_positive"
    for name in ("K3.01", "K4.01", "K5.01", "K6.01"):
        assert pd.isna(out.loc[name, "label"])
    assert out.loc["K3.01", "exclusion_reason"] == "unresolved_candidate"
    assert out.loc["K4.01", "exclusion_reason"] == "disposition_conflict"
    assert out.loc["K5.01", "exclusion_reason"] == "disposition_conflict"
    assert out.loc["K6.01", "exclusion_reason"] == "missing_disposition"
    # Original dispositions are preserved alongside the derived label.
    assert out.loc["K4.01", "koi_disposition"] == "CONFIRMED"
    assert (out["label_rule"] == "dr25_conservative_v1").all()


def test_label_mapping_requires_columns():
    with pytest.raises(KeyError):
        assign_labels(pd.DataFrame({"koi_disposition": ["CONFIRMED"]}))


def _big_catalog(n=200) -> pd.DataFrame:
    rows = [
        (
            f"K{i:05d}.01",
            i,
            "CONFIRMED" if i % 3 else "FALSE POSITIVE",
            "CANDIDATE" if i % 3 else "FALSE POSITIVE",
        )
        for i in range(n)
    ]
    return assign_labels(
        pd.DataFrame(rows, columns=["kepoi_name", "kepid", "koi_disposition", "koi_pdisposition"])
    )


def test_select_subset_is_deterministic_and_order_independent():
    cat = _big_catalog()
    sel = SelectionConfig(seed=7, limit=10)
    a = select_subset(cat, sel)
    b = select_subset(cat.sample(frac=1, random_state=1), sel)
    pd.testing.assert_frame_equal(a, b)
    c = select_subset(cat, SelectionConfig(seed=8, limit=10))
    assert list(a["kepoi_name"]) != list(c["kepoi_name"])


def test_select_subset_is_stratified_and_excludes_unlabelled():
    cat = pd.concat([_big_catalog(), assign_labels(_catalog())])
    out = select_subset(cat, SelectionConfig(seed=1, limit=10))
    assert out["label"].notna().all()
    assert (out["label"] == 1).sum() == 5 and (out["label"] == 0).sum() == 5


def test_allocate_respects_group_sizes():
    assert _allocate(10, [100, 100]) == [5, 5]
    assert _allocate(5, [100, 100]) == [3, 2]
    assert _allocate(10, [2, 100]) == [2, 8]
    assert _allocate(10, [2, 3]) == [2, 3]


def test_ephemeris_from_row_handles_missing():
    assert ephemeris_from_row(pd.Series({"koi_period": float("nan"), "koi_time0bk": 1.0})) is None
    eph = ephemeris_from_row(
        pd.Series({"koi_period": 2.0, "koi_time0bk": 131.0, "koi_duration": 3.0})
    )
    assert eph is not None and eph.period_days == 2.0 and eph.duration_hours == 3.0


def test_build_select_query_validates_identifiers():
    q = build_select_query("q1_q17_dr25_koi", ["kepid", "koi_period"], order_by="kepid")
    assert q == "select kepid,koi_period from q1_q17_dr25_koi order by kepid"
    with pytest.raises(ValueError):
        build_select_query("t; drop table x", ["kepid"])


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_run_tap_query_parses_csv_and_sends_query():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = request.url.params["query"]
        seen["format"] = request.url.params["format"]
        return httpx.Response(200, text="kepid,koi_period\n1,2.5\n")

    df = run_tap_query("select kepid from t", client=_mock_client(handler))
    assert seen == {"query": "select kepid from t", "format": "csv"}
    assert df.to_dict("records") == [{"kepid": 1, "koi_period": 2.5}]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, text="boom"), httpx.Response(200, text="<VOTABLE>ERROR</VOTABLE>")],
)
def test_run_tap_query_raises_on_errors(response):
    with pytest.raises(ArchiveError):
        run_tap_query("select 1", client=_mock_client(lambda r: response))


def test_fetch_catalog_caches_snapshot_with_provenance(tmp_paths):
    calls = {"n": 0}
    header = ",".join(DR25_KOI_COLUMNS)
    row = ",".join("1" if c == "kepid" else "" for c in DR25_KOI_COLUMNS)

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text=f"{header}\n{row}\n")

    client = _mock_client(handler)
    first = fetch_dr25_koi_catalog(tmp_paths, client=client)
    second = fetch_dr25_koi_catalog(tmp_paths, client=client)
    assert calls["n"] == 1
    assert not first.from_cache and second.from_cache
    assert first.meta["table"] == "q1_q17_dr25_koi" and "select" in first.meta["query"]
    assert first.meta["sha256"] == second.meta["sha256"]

    # Tampering with the snapshot is detected.
    catalog_snapshot_path(tmp_paths, "q1_q17_dr25_koi").write_text("kepid\n2\n")
    with pytest.raises(ArchiveError):
        fetch_dr25_koi_catalog(tmp_paths, client=client)


def test_fetch_catalog_rejects_missing_columns(tmp_paths):
    client = _mock_client(lambda r: httpx.Response(200, text="kepid\n1\n"))
    with pytest.raises(ArchiveError, match="missing expected columns"):
        fetch_dr25_koi_catalog(tmp_paths, client=client)


def test_kic_dirname():
    assert kic_dirname(10811496) == "kic_010811496"
