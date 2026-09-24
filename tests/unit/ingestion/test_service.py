"""Ingestion service tests."""

from pathlib import Path

import pytest

from app.ingestion.errors import DocumentExtractionError, UnsupportedFileTypeError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.models import ExtractedSection
from app.ingestion.service import IngestionService


class FailingLoader(DocumentLoader):
    def load(self, file_path: Path) -> list[ExtractedSection]:
        raise RuntimeError("extraction failed")


def test_service_selects_loader_and_returns_normalized_document(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("  Hello   world.\n\nSecond line.  ", encoding="utf-8")
    service = IngestionService()

    document = service.ingest(source)

    assert document.file_type == "txt"
    assert document.file_name == "document.txt"
    assert document.text == "Hello world.\n\nSecond line."
    assert document.metadata.file_size == source.stat().st_size
    assert document.metadata.content_type == "text/plain"
    assert document.document_id == service.ingest(source).document_id


def test_service_rejects_unsupported_format(tmp_path: Path) -> None:
    source = tmp_path / "document.csv"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        IngestionService().ingest(source)


def test_service_reports_extraction_failure(tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(DocumentExtractionError):
        IngestionService(loaders={"txt": FailingLoader()}).ingest(source)