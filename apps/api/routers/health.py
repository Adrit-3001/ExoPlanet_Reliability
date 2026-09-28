from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_repository
from apps.api.schemas.targets import HealthResponse
from exoreliability import __version__
from exoreliability.targets import TargetRepository

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(repo: TargetRepository = Depends(get_repository)) -> HealthResponse:
    return HealthResponse(
        version=__version__,
        catalog_snapshot_available=repo.catalog() is not None,
        processed_targets=len(repo.processed_kepids()),
    )
