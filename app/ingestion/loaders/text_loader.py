"""UTF-8 text extraction."""

from pathlib import Path

from app.ingestion.errors import DocumentExtractionError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.models import ExtractedSection


class TextLoader(DocumentLoader):
    """Read a UTF-8 text file as one logical section."""

    def load(self, file_path: Path) -> list[ExtractedSection]:
        try:
            return [ExtractedSection(text=file_path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeError) as error:
            raise DocumentExtractionError("Unable to read the UTF-8 text file.") from error