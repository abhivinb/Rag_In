"""PDF text extraction."""

from pathlib import Path

from pypdf import PdfReader

from app.ingestion.errors import DocumentExtractionError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.models import ExtractedSection


class PDFLoader(DocumentLoader):
    """Extract text from each PDF page while retaining page numbers."""

    def load(self, file_path: Path) -> list[ExtractedSection]:
        try:
            reader = PdfReader(str(file_path))
            return [
                ExtractedSection(text=page.extract_text() or "", page_number=page_number)
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as error:
            raise DocumentExtractionError("Unable to extract text from the PDF.") from error