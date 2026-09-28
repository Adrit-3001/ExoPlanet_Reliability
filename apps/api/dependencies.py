"""FastAPI dependencies. Tests override ``get_repository`` to point at fixture data."""

from __future__ import annotations

from functools import lru_cache

from exoreliability.config import get_paths
from exoreliability.targets import TargetRepository


@lru_cache(maxsize=1)
def get_repository() -> TargetRepository:
    return TargetRepository(get_paths())
