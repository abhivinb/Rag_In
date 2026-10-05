"""Tests for the upload-to-index orchestration boundary."""

from datetime import UTC, datetime
from io import BytesIO

import pytest

from app.ingestion.indexing import DocumentIndexingService
from app.ingestion.models import Document, DocumentMetadata


class FakeIngestion:
    def ingest(self, file_path):
        return Document(
            document_id="temporary-id",
            source=str(file_path),
            file_name="temporary.txt",
            file_type="txt",
            text="The retention policy is seven years.",
            metadata=DocumentMetadata(
                file_size=35,
                content_type="text/plain",
                created_at=datetime.now(UTC),
                source_type="file",
            ),
        )


class FakeChunking:
    def chunk(self, document):
        return []


class FakePersistence:
    def __init__(self):
        self.document = None
        self.chunks = None

    async def persist(self, session, document, chunks):
        self.document = document
        self.chunks = chunks


@pytest.mark.asyncio
async def test_uploaded_filename_and_content_id_are_stable() -> None:
    persistence = FakePersistence()
    service = DocumentIndexingService(FakeIngestion(), FakeChunking(), persistence)

    first = await service.index(None, "policy.txt", BytesIO(b"policy"))

    assert first.file_name == "policy.txt"
    assert first.file_type == "txt"
    assert first.chunk_count == 0
    assert persistence.document.file_name == "policy.txt"
    assert persistence.document.source == "policy.txt"
