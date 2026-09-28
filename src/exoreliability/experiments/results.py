"""Self-describing experiment run folders under ``artifacts/experiments/``.

Each run folder contains ``config.yaml`` (the validated config as run), ``environment.json``
(software/hardware/git context) and the run's result files.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from exoreliability.config import ProjectPaths
from exoreliability.data.cache import write_json_atomic
from exoreliability.training.reproducibility import environment_info

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def create_run_dir(paths: ProjectPaths, name: str, *, now: datetime | None = None) -> Path:
    """Create ``artifacts/experiments/<UTC timestamp>_<name>/`` (never reusing an existing folder)."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid run name {name!r}")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H%M%SZ")
    base = paths.experiments / f"{stamp}_{name}"
    run_dir, i = base, 1
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{i}")
        i += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_run_metadata(
    run_dir: Path,
    *,
    config: dict[str, Any],
    config_source: Path | None,
    paths: ProjectPaths,
    extra: dict[str, Any] | None = None,
) -> None:
    doc = {"config_source": str(config_source) if config_source else None, **config}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    write_json_atomic(
        run_dir / "environment.json", {**environment_info(paths.root), **(extra or {})}
    )
