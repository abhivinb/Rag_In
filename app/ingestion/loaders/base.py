"""Loader abstraction for supported document formats."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.models import ExtractedSection


class DocumentLoader(ABC):
    """Extract document sections without format-specific service logic."""

    @abstractmethod
    def load(self, file_path: Path) -> list[ExtractedSection]:
        """Extract ordered sections from a validated file."""