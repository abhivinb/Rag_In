"""DOCX paragraph extraction."""

from pathlib import Path

from docx import Document as DOCXDocument

from app.ingestion.errors import DocumentExtractionError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.models import ExtractedSection


class DOCXLoader(DocumentLoader):
    """Extract non-empty paragraphs in their source order."""

    def load(self, file_path: Path) -> list[ExtractedSection]:
        try:
            document = DOCXDocument(str(file_path))
            return [
                ExtractedSection(text=paragraph.text)
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
        except Exception as error:
            raise DocumentExtractionError("Unable to extract text from the DOCX document.") from error