"""Centralized logging setup."""

import logging

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure consistent application-wide logging."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
