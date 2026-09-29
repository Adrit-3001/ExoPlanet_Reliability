"""Label policy dr25_clean_v2 on SYNTHETIC catalog rows."""

from __future__ import annotations

import pandas as pd
import pytest

from exoreliability.data.labels import (
    LABEL_POLICY,
    POLICY_TABLE,
    UnexpectedDispositionError,
    apply_label_policy,
    decide,
)

COLS = ["kepoi_name", "kepid", "koi_disposition", "koi_pdisposition"]


def _cat(rows):
    return pd.DataFrame(rows, columns=COLS)


@pytest.mark.parametrize(
    ("archive", "kepler", "label", "reason"),
    [
        ("CONFIRMED", "CANDIDATE", 1, ""),
        ("FALSE POSITIVE", "FALSE POSITIVE", 0, ""),
        ("CANDIDATE", "CANDIDATE", None, "unresolved_candidate"),
        ("CONFIRMED", "FALSE POSITIVE", None, "disposition_conflict"),
        ("FALSE POSITIVE", "CANDIDATE", None, "disposition_conflict"),
        ("CANDIDATE", "FALSE POSITIVE", None, "disposition_conflict"),
        ("NOT DISPOSITIONED", "NOT DISPOSITIONED", None, "not_dispositioned"),
        ("CONFIRMED", "NOT DISPOSITIONED", None, "not_dispositioned"),
        (None, "CANDIDATE", None, "missing_disposition"),
        ("CONFIRMED", float("nan"), None, "missing_disposition"),
        ("  confirmed ", "candidate", 1, ""),  # case/whitespace are normalised for matching only
    ],
)
def test_decision_table(archive, kepler, label, reason):
    d = decide(archive, kepler)
    assert d.label == label
    assert d.reason == reason


@pytest.mark.parametrize(
    ("archive", "kepler"),
    [("PLANET", "CANDIDATE"), ("CONFIRMED", "CONFIRMED"), ("FP", "FALSE POSITIVE")],
)
def test_unexpected_values_raise(archive, kepler):
    with pytest.raises(UnexpectedDispositionError):
        decide(archive, kepler)
    with pytest.raises(UnexpectedDispositionError):
        apply_label_policy(_cat([("K1.01", 1, archive, kepler)]))


def test_unexpected_values_can_be_explicitly_excluded():
    out = apply_label_policy(_cat([("K1.01", 1, "PLANET", "CANDIDATE")]), on_unexpected="exclude")
    assert pd.isna(out.loc[0, "label"]) and out.loc[0, "exclusion_reason"] == "unexpected_value"


def test_policy_columns_and_original_values_preserved():
    rows = [
        ("K1.01", 1, "CONFIRMED", "CANDIDATE"),
        ("K2.01", 2, "FALSE POSITIVE", "FALSE POSITIVE"),
        ("K3.01", 3, "CANDIDATE", "CANDIDATE"),
        ("K4.01", 4, "CONFIRMED", "FALSE POSITIVE"),
    ]
    original = _cat(rows)
    out = apply_label_policy(original)
    pd.testing.assert_frame_equal(out[COLS], original)  # raw columns untouched
    assert out["label"].iloc[:2].tolist() == [1, 0] and out["label"].iloc[2:].isna().all()
    assert list(out["training_status"]) == [
        "train_eligible",
        "train_eligible",
        "excluded",
        "excluded",
    ]
    assert list(out["label_name"]) == ["planet", "false_positive", "", ""]
    assert (out["label_policy"] == LABEL_POLICY).all()
    assert "label" not in original.columns  # input not mutated


def test_candidates_are_kept_not_dropped():
    out = apply_label_policy(_cat([("K3.01", 3, "CANDIDATE", "CANDIDATE")]))
    assert len(out) == 1 and out.loc[0, "training_status"] == "excluded"


def test_every_documented_pair_is_covered():
    archive = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "NOT DISPOSITIONED"]
    kepler = ["CANDIDATE", "FALSE POSITIVE", "NOT DISPOSITIONED"]
    for a in archive:
        for k in kepler:
            decide(a, k)  # must not raise KeyError
    assert ("CONFIRMED", "CANDIDATE") in POLICY_TABLE


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        apply_label_policy(pd.DataFrame({"koi_disposition": ["CONFIRMED"]}))
