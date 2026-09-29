"""Supervised label policy for DR25 KOIs.

Policy ``dr25_clean_v2`` (see ``docs/data_contract.md`` §2). Every combination of the two
disposition columns maps to exactly one row of ``POLICY_TABLE``; any value outside the
documented vocabulary raises ``UnexpectedDispositionError`` instead of being mapped.

Columns used (verified against the archive's ``TAP_SCHEMA`` and column documentation):

* ``koi_disposition`` — "Exoplanet Archive Disposition": CANDIDATE, FALSE POSITIVE,
  NOT DISPOSITIONED or CONFIRMED. CONFIRMED means the object is in the archive's
  Confirmed Planets table (confirmation may rely on non-Kepler follow-up).
* ``koi_pdisposition`` — "Disposition Using Kepler Data": the DR25 Robovetter outcome,
  CANDIDATE, FALSE POSITIVE or NOT DISPOSITIONED.

The original columns are never modified; derived columns are added alongside them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

LABEL_POLICY = "dr25_clean_v2"

POSITIVE = 1
NEGATIVE = 0

ARCHIVE_VALUES = frozenset({"CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "NOT DISPOSITIONED"})
KEPLER_VALUES = frozenset({"CANDIDATE", "FALSE POSITIVE", "NOT DISPOSITIONED"})


class UnexpectedDispositionError(ValueError):
    """A disposition value outside the documented archive vocabulary."""


@dataclass(frozen=True)
class LabelDecision:
    label: int | None
    label_name: str  # "planet", "false_positive" or "" when excluded
    reason: str  # "" for training-eligible rows


_POS = LabelDecision(POSITIVE, "planet", "")
_NEG = LabelDecision(NEGATIVE, "false_positive", "")

# (koi_disposition, koi_pdisposition) -> decision. NOT DISPOSITIONED and missing values are
# handled before this lookup; every remaining documented combination is listed explicitly.
POLICY_TABLE: Mapping[tuple[str, str], LabelDecision] = {
    ("CONFIRMED", "CANDIDATE"): _POS,
    ("FALSE POSITIVE", "FALSE POSITIVE"): _NEG,
    ("CANDIDATE", "CANDIDATE"): LabelDecision(None, "", "unresolved_candidate"),
    ("CONFIRMED", "FALSE POSITIVE"): LabelDecision(None, "", "disposition_conflict"),
    ("FALSE POSITIVE", "CANDIDATE"): LabelDecision(None, "", "disposition_conflict"),
    ("CANDIDATE", "FALSE POSITIVE"): LabelDecision(None, "", "disposition_conflict"),
}


def _normalise(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    text = " ".join(str(value).split()).upper()
    return text or None


def decide(archive: object, kepler: object) -> LabelDecision:
    """Label decision for one KOI. Raises ``UnexpectedDispositionError`` for unknown values."""
    a, k = _normalise(archive), _normalise(kepler)
    if a is None or k is None:
        return LabelDecision(None, "", "missing_disposition")
    if a not in ARCHIVE_VALUES:
        raise UnexpectedDispositionError(f"unexpected koi_disposition {archive!r}")
    if k not in KEPLER_VALUES:
        raise UnexpectedDispositionError(f"unexpected koi_pdisposition {kepler!r}")
    if "NOT DISPOSITIONED" in (a, k):
        return LabelDecision(None, "", "not_dispositioned")
    return POLICY_TABLE[(a, k)]


def apply_label_policy(
    catalog: pd.DataFrame,
    *,
    on_unexpected: Literal["raise", "exclude"] = "raise",
) -> pd.DataFrame:
    """Return a copy of ``catalog`` with derived label columns added.

    Added columns: ``label`` (nullable Int8), ``label_name``, ``training_status``
    (``train_eligible`` / ``excluded``), ``exclusion_reason`` and ``label_policy``.
    With ``on_unexpected="exclude"`` unknown values are kept but excluded with reason
    ``unexpected_value``; the default raises.
    """
    required = {"koi_disposition", "koi_pdisposition"}
    if not required <= set(catalog.columns):
        raise KeyError(
            f"catalog lacks disposition columns {sorted(required - set(catalog.columns))}"
        )

    decisions = []
    for a, k in zip(catalog["koi_disposition"], catalog["koi_pdisposition"], strict=True):
        try:
            decisions.append(decide(a, k))
        except UnexpectedDispositionError:
            if on_unexpected == "raise":
                raise
            decisions.append(LabelDecision(None, "", "unexpected_value"))

    out = catalog.copy()
    out["label"] = pd.array([d.label for d in decisions], dtype="Int8")
    out["label_name"] = [d.label_name for d in decisions]
    out["training_status"] = [
        "train_eligible" if d.label is not None else "excluded" for d in decisions
    ]
    out["exclusion_reason"] = [d.reason for d in decisions]
    out["label_policy"] = LABEL_POLICY
    return out
