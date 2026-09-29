"""KIC-grouped split tests on a SYNTHETIC catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoreliability.config import SplitConfig
from exoreliability.data.labels import apply_label_policy
from exoreliability.data.splits import (
    SPLITS,
    LeakageError,
    SplitsExistError,
    _cut_sizes,
    build_split_manifest,
    check_no_leakage,
    load_splits,
    star_strata,
    write_splits,
)

CATALOG_META = {
    "table": "synthetic",
    "query": "n/a",
    "retrieved_at_utc": "2000-01-01",
    "sha256": "x",
    "n_rows": 0,
}


def synthetic_catalog(n_stars: int = 600, seed: int = 0) -> pd.DataFrame:
    """Stars with 1–4 KOIs each; multi-KOI stars skew to confirmed planets (as in DR25)."""
    rng = np.random.default_rng(seed)
    rows = []
    for kepid in range(1000, 1000 + n_stars):
        n = int(rng.choice([1, 1, 1, 1, 2, 3, 4]))
        for j in range(n):
            r = rng.random()
            if (n > 1 and r < 0.7) or r < 0.35:
                disp = ("CONFIRMED", "CANDIDATE")
            elif r < 0.85:
                disp = ("FALSE POSITIVE", "FALSE POSITIVE")
            else:
                disp = ("CANDIDATE", "CANDIDATE")
            rows.append((kepid, f"K{kepid:05d}.{j + 1:02d}", *disp))
    return apply_label_policy(
        pd.DataFrame(rows, columns=["kepid", "kepoi_name", "koi_disposition", "koi_pdisposition"])
    )


@pytest.fixture(scope="module")
def catalog():
    return synthetic_catalog()


def test_no_kic_overlap_between_splits(catalog):
    manifest = build_split_manifest(catalog, SplitConfig())
    result = check_no_leakage(manifest)
    assert result == {
        "shared_kic_train_val": 0,
        "shared_kic_train_test": 0,
        "shared_kic_val_test": 0,
        "duplicate_koi_ids": 0,
    }
    # All KOIs of one star share its split (including excluded candidates).
    assert (manifest.groupby("kepid").split.nunique() == 1).all()
    assert len(manifest) == len(catalog)


def test_deterministic_and_order_independent(catalog):
    a = build_split_manifest(catalog, SplitConfig(seed=7))
    b = build_split_manifest(catalog.sample(frac=1, random_state=3), SplitConfig(seed=7))
    pd.testing.assert_frame_equal(a, b)
    c = build_split_manifest(catalog, SplitConfig(seed=8))
    assert not a[["kepid", "split"]].equals(c[["kepid", "split"]])


def test_star_level_fractions_and_label_balance(catalog):
    manifest = build_split_manifest(catalog, SplitConfig())
    stars = manifest.drop_duplicates("kepid")
    frac = stars.split.value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.02)
    assert frac["val"] == pytest.approx(0.15, abs=0.02)
    assert frac["test"] == pytest.approx(0.15, abs=0.02)
    eligible = manifest[manifest.training_status == "train_eligible"]
    overall = (eligible.label == 1).mean()
    for s in SPLITS:
        assert (eligible[eligible.split == s].label == 1).mean() == pytest.approx(overall, abs=0.06)


def test_strata():
    cat = apply_label_policy(
        pd.DataFrame(
            [
                (1, "K1.01", "CONFIRMED", "CANDIDATE"),
                (2, "K2.01", "FALSE POSITIVE", "FALSE POSITIVE"),
                (3, "K3.01", "CONFIRMED", "CANDIDATE"),
                (3, "K3.02", "FALSE POSITIVE", "FALSE POSITIVE"),
                (4, "K4.01", "CANDIDATE", "CANDIDATE"),
            ],
            columns=["kepid", "kepoi_name", "koi_disposition", "koi_pdisposition"],
        )
    )
    s = star_strata(cat).set_index("kepid").star_stratum.to_dict()
    assert s == {1: "pos_only", 2: "neg_only", 3: "mixed", 4: "unlabelled_only"}


@pytest.mark.parametrize("n", [0, 1, 2, 7, 10, 101])
def test_cut_sizes_sum(n):
    sizes = _cut_sizes(n, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert sum(sizes.values()) == n


def test_write_requires_overwrite_and_reload_is_identical(catalog, tmp_path):
    cfg = SplitConfig()
    manifest = build_split_manifest(catalog, cfg)
    meta = write_splits(manifest, tmp_path, cfg=cfg, catalog_meta=CATALOG_META)
    assert meta["leakage_checks"]["shared_kic_train_test"] == 0
    assert meta["n_kics"] == catalog.kepid.nunique()

    with pytest.raises(SplitsExistError):
        write_splits(manifest, tmp_path, cfg=cfg, catalog_meta=CATALOG_META)
    write_splits(manifest, tmp_path, cfg=cfg, catalog_meta=CATALOG_META, overwrite=True)

    loaded = load_splits(tmp_path)
    left = manifest.sort_values("kepoi_name").reset_index(drop=True)
    right = loaded.manifest.sort_values("kepoi_name").reset_index(drop=True)[left.columns]
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_load_detects_tampering(catalog, tmp_path):
    cfg = SplitConfig()
    write_splits(build_split_manifest(catalog, cfg), tmp_path, cfg=cfg, catalog_meta=CATALOG_META)
    with (tmp_path / "val.csv").open("a") as fh:
        fh.write(
            "1000,K01000.99,val,pos_only,1,planet,train_eligible,,dr25_clean_v2,CONFIRMED,CANDIDATE\n"
        )
    with pytest.raises(ValueError, match="checksum"):
        load_splits(tmp_path)


def test_leakage_check_detects_shared_star():
    bad = pd.DataFrame(
        {"kepid": [1, 1], "kepoi_name": ["K1.01", "K1.02"], "split": ["train", "test"]}
    )
    with pytest.raises(LeakageError):
        check_no_leakage(bad)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitConfig(train_fraction=0.8, val_fraction=0.15, test_fraction=0.15)


# -- star selection for dataset builds --------------------------------------------------


def _split_set(catalog, tmp_path):
    cfg = SplitConfig()
    write_splits(build_split_manifest(catalog, cfg), tmp_path, cfg=cfg, catalog_meta=CATALOG_META)
    return load_splits(tmp_path)


def test_select_stars_deterministic_balanced_and_respects_splits(catalog, tmp_path):
    from exoreliability.config import StarSelectionConfig
    from exoreliability.data.selection import select_stars

    splits = _split_set(catalog, tmp_path)
    anchor = int(splits.manifest.kepid.iloc[0])
    sel_cfg = StarSelectionConfig(seed=3, n_stars=40, include_kepids=[anchor])
    a = select_stars(splits, sel_cfg)
    b = select_stars(splits, sel_cfg)
    pd.testing.assert_frame_equal(a, b)
    assert anchor in set(a.kepid) and len(a) == 41
    split_of = splits.split_of()
    assert all(
        split_of[k] == s for k, s in zip(a.kepid, a.split, strict=True)
    )  # pre-assigned split kept
    sampled = a[a.selection_reason == "sampled"]
    assert sampled.split.value_counts().to_dict() == {"train": 28, "val": 6, "test": 6}
    assert set(sampled.star_stratum) <= {
        "pos_only",
        "neg_only",
        "mixed",
    }  # never unlabelled-only stars
    has_pos = sampled.star_stratum.isin(["pos_only", "mixed"])
    assert has_pos.sum() == 20 and (~has_pos).sum() == 20


def test_select_stars_rejects_unknown_anchor(catalog, tmp_path):
    from exoreliability.config import StarSelectionConfig
    from exoreliability.data.selection import select_stars

    with pytest.raises(ValueError):
        select_stars(_split_set(catalog, tmp_path), StarSelectionConfig(include_kepids=[1]))
