"""Python logging setup. Use this instead of print() everywhere."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "geoconstruct",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or return a configured logger.

    Adds a stdout handler and optionally a file handler. Idempotent: calling
    twice with the same name does not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
