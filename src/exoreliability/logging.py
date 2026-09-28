"""Logging setup shared by scripts and the API. Library modules only call ``getLogger``."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, honouring ``EXORELIABILITY_LOG_LEVEL``."""
    level_name = (level or os.environ.get("EXORELIABILITY_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Do not touch the "astroquery"/"astropy" loggers here: creating them before astropy is
    # imported yields plain Loggers instead of AstropyLogger and breaks astroquery.
    for noisy in ("lightkurve", "matplotlib", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
