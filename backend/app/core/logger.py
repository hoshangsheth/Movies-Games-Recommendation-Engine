"""
Application-wide logger configuration.

The original Streamlit app used bare `print()` statements (e.g. "Best
Match is: ..."). This module replaces that with a standard configured
Python logger so log output is consistent across the API layer, services,
and the recommendation engine, and can be wired into any log aggregator.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once at application startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. under uvicorn's --reload).
    if not root_logger.handlers:
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger. Use `get_logger(__name__)` at the top of a module."""
    return logging.getLogger(name)
