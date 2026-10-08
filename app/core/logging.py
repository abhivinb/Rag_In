"""Centralized logging setup."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure consistent application-wide logging."""
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=settings.log_level.upper(),
        handlers=[console_handler, file_handler],
        force=True,
    )
