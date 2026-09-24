"""Chunk model tests."""

from datetime import UTC, datetime

from app.ingestion.chunking.models import ChunkMetadata, DocumentChunk


def test_document_chunk_model_contains_embedding_ready_fields() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-id",
        document_id="document-id",
        content="content",
        chunk_index=0,
        metadata=ChunkMetadata(
            source="data/raw/example.txt",
            file_name="example.txt",
            file_type="txt",
            source_type="file",
            chunk_start=0,
            chunk_end=7,
        ),
    )

    assert chunk.document_id == "document-id"
    assert chunk.metadata.chunk_end == 7
