"""Orchestration for the document ingestion pipeline."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from app.ingestion.cleaner import clean_text
from app.ingestion.errors import DocumentExtractionError, IngestionError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import TextLoader
from app.ingestion.models import Document, DocumentMetadata
from app.ingestion.validators import CONTENT_TYPES, validate_file

logger = logging.getLogger(__name__)


class IngestionService:
    """Validate, extract, clean, and normalize supported source documents."""

    def __init__(self, loaders: Mapping[str, DocumentLoader] | None = None) -> None:
        self._loaders = dict(
            loaders
            or {"pdf": PDFLoader(), "docx": DOCXLoader(), "txt": TextLoader()}
        )

    def ingest(self, file_path: str | Path) -> Document:
        """Ingest one source file and return its normalized representation."""
        logger.info("Document ingestion started")
        try:
            path, file_type = validate_file(file_path)
            logger.info("Document file type detected: %s", file_type)
            loader = self._loaders[file_type]
            sections = loader.load(path)
            cleaned_sections = [clean_text(section.text) for section in sections]
            text = "\n\n".join(section for section in cleaned_sections if section)
            file_size = path.stat().st_size
            document = Document.from_path(
                document_id=_document_id(path, text),
                source=path,
                file_type=file_type,
                text=text,
                metadata=DocumentMetadata(
                    page_number=None,
                    page_numbers=[
                        section.page_number
                        for section in sections
                        if section.page_number is not None
                    ],
                    file_size=file_size,
                    content_type=CONTENT_TYPES[file_type],
                    created_at=datetime.now(UTC),
                    source_type="file",
                ),
            )
            logger.info("Document extraction completed")
            logger.info("Document ingestion completed")
            return document
        except IngestionError:
            logger.exception("Document ingestion failed")
            raise
        except Exception as error:
            logger.exception("Document ingestion failed")
            raise DocumentExtractionError("Document ingestion failed.") from error


def _document_id(path: Path, text: str) -> str:
    """Create a stable identifier from the source path and normalized content."""
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()