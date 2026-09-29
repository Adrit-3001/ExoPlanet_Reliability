"""Read-only summaries of datasets built by scripts/build_dataset.py."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_repository
from apps.api.schemas.datasets import DatasetList, DatasetSummary
from exoreliability.data.cache import read_json
from exoreliability.targets import TargetRepository

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _summary(doc: dict[str, Any]) -> DatasetSummary:
    c = doc["counts"]
    return DatasetSummary(
        name=doc["name"],
        scale=doc["scale"],
        created_at_utc=doc["created_at_utc"],
        n_examples=c["n_examples"],
        n_trainable=c["n_trainable"],
        n_stars_ok=c["n_stars_ok"],
        example_status=c["example_status"],
        trainable_by_split=c["trainable_by_split"],
        shapes=doc["shapes"],
        label_policy=doc.get("label_policy", "unknown"),
        split_name=doc["splits"]["name"],
    )


@router.get("", response_model=DatasetList)
def list_datasets(repo: TargetRepository = Depends(get_repository)) -> DatasetList:
    root = repo.paths.datasets
    docs = sorted(root.glob("*/dataset.json")) if root.exists() else []
    return DatasetList(datasets=[_summary(read_json(p)) for p in docs])


@router.get("/{name}", response_model=DatasetSummary)
def get_dataset(name: str, repo: TargetRepository = Depends(get_repository)) -> DatasetSummary:
    if not name.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=422, detail="invalid dataset name")
    path = repo.paths.datasets / name / "dataset.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset {name!r} not found")
    return _summary(read_json(path))
