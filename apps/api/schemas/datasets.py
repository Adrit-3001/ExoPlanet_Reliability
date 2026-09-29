"""Response contracts for built-dataset summaries."""

from __future__ import annotations

from pydantic import BaseModel


class SplitCounts(BaseModel):
    n: int
    n_positive: int
    n_negative: int
    n_kics: int


class DatasetSummary(BaseModel):
    name: str
    scale: str
    created_at_utc: str
    n_examples: int
    n_trainable: int
    n_stars_ok: int
    example_status: dict[str, int]
    trainable_by_split: dict[str, SplitCounts]
    shapes: dict[str, list[int]]
    label_policy: str
    split_name: str


class DatasetList(BaseModel):
    datasets: list[DatasetSummary]
