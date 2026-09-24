"""Opt-in PostgreSQL/pgvector integration tests."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text

from app.core.config import Settings
from app.database.exceptions import PersistenceError
from app.database.models import ChunkRecord
from app.database.repository import VectorRepository
from app.database.session import create_engine, create_session_factory, enable_pgvector
from app.embeddings.base import EmbeddingProvider
from app.embeddings.models import EmbeddingConfig, EmbeddingResult
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking.models import ChunkMetadata, DocumentChunk
from app.ingestion.models import Document, DocumentMetadata


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run PostgreSQL integration tests.",
)


class FakeIntegrationProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] * 1536 for index, _ in enumerate(texts)]


def make_document() -> Document:
    return Document(
        document_id="integration-phase4-document",
        source="integration.txt",
        file_name="integration.txt",
        file_type="txt",
        text="First integration chunk. Second integration chunk.",
        metadata=DocumentMetadata(
            file_size=50,
            content_type="text/plain",
            created_at=datetime.now(UTC),
            source_type="test",
        ),
    )


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"integration-phase4-chunk-{index}",
            document_id="integration-phase4-document",
            content=content,
            chunk_index=index,
            metadata=ChunkMetadata(
                source="integration.txt",
                file_name="integration.txt",
                file_type="txt",
                source_type="test",
                chunk_start=index * 25,
                chunk_end=(index + 1) * 25,
            ),
        )
        for index, content in enumerate(("First integration chunk.", "Second integration chunk."))
    ]


@pytest.mark.asyncio
async def test_postgres_pgvector_persistence_and_idempotency() -> None:
    settings = Settings()
    engine = create_engine(settings)
    await enable_pgvector(engine)
    session_factory = create_session_factory(engine)
    document = make_document()
    chunks = make_chunks()
    service = EmbeddingService(
        FakeIntegrationProvider(), EmbeddingConfig(dimension=1536, batch_size=2)
    )
    repository = VectorRepository()

    try:
        async with session_factory() as session:
            embeddings = await service.embed_chunks(chunks)
            await repository.persist_document(session, document, chunks, embeddings)
            await repository.persist_document(session, document, chunks, embeddings)

        async with session_factory() as session:
            persisted_document = await repository.get_document(session, document.document_id)
            persisted_chunks = await repository.get_chunks(session, document.document_id)
            count = await session.scalar(
                select(func.count()).select_from(ChunkRecord).where(
                    ChunkRecord.document_id == document.document_id
                )
            )

        assert persisted_document is not None
        assert persisted_document.document_metadata["source_type"] == "test"
        assert len(persisted_chunks) == 2
        assert all(len(chunk.embedding) == 1536 for chunk in persisted_chunks)
        assert count == 2

        async with session_factory() as session:
            await repository.persist_document(session, document, [], [])

        async with session_factory() as session:
            assert await repository.get_document(session, document.document_id) is not None
            assert await repository.get_chunks(session, document.document_id) == []
    finally:
        async with session_factory() as session:
            await session.execute(
                ChunkRecord.__table__.delete().where(
                    ChunkRecord.document_id == document.document_id
                )
            )
            await session.commit()
            await session.execute(
                text("DELETE FROM documents WHERE document_id = :document_id"),
                {"document_id": document.document_id},
            )
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_migrated_schema_and_transaction_rollback() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    document = make_document()
    chunks = make_chunks()
    repository = VectorRepository()
    embeddings = [
        EmbeddingResult(chunk_id=chunk.chunk_id, vector=[0.0] * 1536)
        for chunk in chunks
    ]

    try:
        async with session_factory() as session:
            version = await session.scalar(text("SELECT version_num FROM alembic_version"))
            table_exists = await session.scalar(
                text(
                    "SELECT to_regclass('public.document_chunks') IS NOT NULL"
                )
            )
            assert version == "0001_initial_vector_store"
            assert table_exists is True
            await session.rollback()

        async with session_factory() as session:
            await repository.persist_document(session, document, chunks, embeddings)

        changed_document = document.model_copy(update={"source": "changed.txt"})
        bad_embeddings = [
            EmbeddingResult(chunk_id=chunks[0].chunk_id, vector=[0.0])
        ]
        async with session_factory() as session:
            with pytest.raises(PersistenceError):
                await repository.persist_document(
                    session, changed_document, chunks[:1], bad_embeddings
                )

        async with session_factory() as session:
            persisted = await repository.get_document(session, document.document_id)
            persisted_chunks = await repository.get_chunks(session, document.document_id)
            assert persisted is not None
            assert persisted.source == document.source
            assert len(persisted_chunks) == 2
    finally:
        async with session_factory() as session:
            await session.execute(
                ChunkRecord.__table__.delete().where(
                    ChunkRecord.document_id == document.document_id
                )
            )
            await session.commit()
            await session.execute(
                text("DELETE FROM documents WHERE document_id = :document_id"),
                {"document_id": document.document_id},
            )
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_foreign_key_rejects_orphan_chunk() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    repository = VectorRepository()
    orphan = make_chunks()[0].model_copy(update={"document_id": "missing-document"})
    embedding = EmbeddingResult(chunk_id=orphan.chunk_id, vector=[0.0] * 1536)

    try:
        async with session_factory() as session:
            with pytest.raises(PersistenceError):
                await repository.upsert_chunks(session, [orphan], [embedding])
            await session.rollback()
    finally:
        await engine.dispose()