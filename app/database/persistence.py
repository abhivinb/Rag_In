"""Embedding and vector persistence orchestration."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repository import VectorRepository
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking.models import DocumentChunk
from app.ingestion.models import Document


class EmbeddingPersistenceService:
    """Generate embeddings and persist a document transactionally."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        repository: VectorRepository,
    ) -> None:
        self.embedding_service = embedding_service
        self.repository = repository

    async def persist(
        self,
        session: AsyncSession,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """Embed chunks and persist document, chunks, and vectors."""
        embeddings = await self.embedding_service.embed_chunks(chunks)
        await self.repository.persist_document(session, document, chunks, embeddings)