"""Transactional document and vector repository."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.exceptions import PersistenceError
from app.database.models import ChunkRecord, DocumentRecord
from app.embeddings.models import EmbeddingResult
from app.ingestion.chunking.models import DocumentChunk
from app.ingestion.models import Document


class VectorRepository:
    """Persist documents and embeddings with idempotent PostgreSQL upserts."""

    async def upsert_document(
        self, session: AsyncSession, document: Document
    ) -> None:
        """Insert or update one document record."""
        now = datetime.now(UTC)
        values = {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "source": document.source,
            "source_type": document.metadata.source_type,
            "metadata": document.metadata.model_dump(mode="json"),
            "updated_at": now,
        }
        table = DocumentRecord.__table__
        statement = insert(table).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[table.c.document_id],
            set_={key: statement.excluded[key] for key in values if key != "document_id"},
        )
        try:
            await session.execute(statement)
        except Exception as error:
            raise PersistenceError("Unable to persist document metadata.") from error

    async def upsert_chunks(
        self,
        session: AsyncSession,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
        *,
        document_id: str | None = None,
    ) -> None:
        """Insert or update chunks and remove stale chunks for the same document."""
        if len(chunks) != len(embeddings):
            raise PersistenceError("Chunk and embedding counts do not match.")
        embedding_by_id = {result.chunk_id: result.vector for result in embeddings}
        if len(embedding_by_id) != len(embeddings):
            raise PersistenceError("Embedding chunk IDs must be unique.")
        if any(chunk.chunk_id not in embedding_by_id for chunk in chunks):
            raise PersistenceError("Every chunk must have a matching embedding.")
        if not chunks:
            if document_id is None:
                raise PersistenceError(
                    "document_id is required when persisting zero chunks."
                )
            try:
                await session.execute(
                    delete(ChunkRecord).where(ChunkRecord.document_id == document_id)
                )
            except Exception as error:
                raise PersistenceError("Unable to remove stale document chunks.") from error
            return
        document_id = chunks[0].document_id
        if any(chunk.document_id != document_id for chunk in chunks):
            raise PersistenceError("Chunks from multiple documents cannot share a transaction.")
        values = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(mode="json"),
                "embedding": embedding_by_id[chunk.chunk_id],
                "updated_at": datetime.now(UTC),
            }
            for chunk in chunks
        ]
        table = ChunkRecord.__table__
        statement = insert(table).values(values)
        update_fields = {
            key: getattr(statement.excluded, key)
            for key in (
                "document_id",
                "chunk_index",
                "content",
                "metadata",
                "embedding",
                "updated_at",
            )
        }
        statement = statement.on_conflict_do_update(
            index_elements=[table.c.chunk_id], set_=update_fields
        )
        try:
            await session.execute(statement)
            await session.execute(
                delete(ChunkRecord).where(
                    ChunkRecord.document_id == document_id,
                    ChunkRecord.chunk_id.not_in([chunk.chunk_id for chunk in chunks]),
                )
            )
        except Exception as error:
            raise PersistenceError("Unable to persist document chunks.") from error

    async def persist_document(
        self,
        session: AsyncSession,
        document: Document,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> None:
        """Persist a document and all chunks in the caller's transaction."""
        try:
            async with session.begin():
                await self.upsert_document(session, document)
                await self.upsert_chunks(
                    session, chunks, embeddings, document_id=document.document_id
                )
        except PersistenceError:
            raise
        except Exception as error:
            raise PersistenceError("Document persistence transaction failed.") from error

    async def get_document(
        self, session: AsyncSession, document_id: str
    ) -> DocumentRecord | None:
        """Retrieve a document by its stable identifier."""
        return await session.scalar(
            select(DocumentRecord).where(DocumentRecord.document_id == document_id)
        )

    async def get_chunks(
        self, session: AsyncSession, document_id: str
    ) -> list[ChunkRecord]:
        """Retrieve chunks in their persisted source order."""
        result = await session.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.chunk_index)
        )
        return list(result)