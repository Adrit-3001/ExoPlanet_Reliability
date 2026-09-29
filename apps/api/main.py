"""FastAPI application. Run from the repository root:

uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import datasets, health, targets
from exoreliability import __version__
from exoreliability.logging import setup_logging

setup_logging()

app = FastAPI(
    title="ExoReliability Lab API",
    version=__version__,
    description="Read-only access to locally processed Kepler DR25 targets, light curves and BLS runs.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.environ.get("EXORELIABILITY_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(targets.router)
app.include_router(datasets.router)
