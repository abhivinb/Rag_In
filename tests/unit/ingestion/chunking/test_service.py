"""Chunking service tests."""

from datetime import UTC, datetime

import pytest

from app.ingestion.chunking.base import ChunkingStrategy
from app.ingestion.chunking.models import ChunkMetadata, ChunkingConfig, DocumentChunk
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.chunking.validators import ChunkValidationError
from app.ingestion.chunking.validators import ChunkingConfigurationError
from app.ingestion.models import Document, DocumentMetadata


def make_document() -> Document:
    return Document(
        document_id="document-id",
        source="example.txt",
        file_name="example.txt",
        file_type="txt",
        text="Document text.",
        metadata=DocumentMetadata(
            file_size=15,
            content_type="text/plain",
            created_at=datetime.now(UTC),
            source_type="file",
        ),
    )


class RecordingStrategy(ChunkingStrategy):
    def __init__(self) -> None:
        self.called_with: Document | None = None

    def chunk(self, document: Document) -> list[DocumentChunk]:
        self.called_with = document
        return [
            DocumentChunk(
                chunk_id="chunk-id",
                document_id=document.document_id,
                content=document.text,
                chunk_index=0,
                metadata=ChunkMetadata(
                    source=document.source,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    source_type=document.metadata.source_type,
                    chunk_start=0,
                    chunk_end=len(document.text),
                ),
            )
        ]


class InvalidStrategy(ChunkingStrategy):
    def chunk(self, document: Document) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id="invalid",
                document_id="other-document",
                content="invalid",
                chunk_index=0,
                metadata=ChunkMetadata(
                    source=document.source,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    source_type=document.metadata.source_type,
                    chunk_start=0,
                    chunk_end=7,
                ),
            )
        ]


def test_service_invokes_strategy_and_validates_result() -> None:
    strategy = RecordingStrategy()

    chunks = ChunkingService(strategy=strategy).chunk(make_document())

    assert strategy.called_with is not None
    assert chunks[0].document_id == "document-id"


def test_service_rejects_invalid_strategy_output() -> None:
    with pytest.raises(ChunkValidationError):
        ChunkingService(strategy=InvalidStrategy()).chunk(make_document())


def test_service_rejects_invalid_configuration() -> None:
    with pytest.raises(ChunkingConfigurationError):
        ChunkingService(config=ChunkingConfig(chunk_size=10, chunk_overlap=10))


def test_service_uses_configured_recursive_strategy() -> None:
    chunks = ChunkingService(
        config=ChunkingConfig(chunk_size=5, chunk_overlap=1, minimum_chunk_size=1)
    ).chunk(make_document())

    assert len(chunks) > 1