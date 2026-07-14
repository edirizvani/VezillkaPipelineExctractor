"""
Vezilka v2 — Logging Configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the root logger for the Vezilka pipeline."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    for noisy in ("urllib3", "httpx", "playwright", "httpcore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root = logging.getLogger("vezilka")
    root.info("Logging initialised (level=%s)", level)
    return root
