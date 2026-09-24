"""PostgreSQL/pgvector similarity retrieval repository."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ChunkRecord
from app.retrieval.exceptions import RetrievalDatabaseError
from app.retrieval.filters import apply_retrieval_filters
from app.retrieval.models import RetrievalConfig, RetrievalFilter


class RetrievalRow:
    """Small database-independent row projection used by the service."""

    def __init__(
        self,
        *,
        chunk_id: str,
        document_id: str,
        content: str,
        score: float,
        metadata: dict[str, Any],
        chunk_index: int,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.chunk_index = chunk_index


class VectorRetrievalRepository:
    """Execute database-side cosine similarity retrieval."""

    async def search(
        self,
        session: AsyncSession,
        query_vector: Sequence[float],
        config: RetrievalConfig,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievalRow]:
        """Return thresholded, ranked, limited vector matches."""
        distance = ChunkRecord.embedding.cosine_distance(list(query_vector))
        similarity = (1.0 - distance).label("similarity")
        statement: Select = select(
            ChunkRecord.chunk_id,
            ChunkRecord.document_id,
            ChunkRecord.content,
            ChunkRecord.chunk_metadata,
            ChunkRecord.chunk_index,
            similarity,
        ).where(
            ChunkRecord.embedding.is_not(None),
            similarity >= config.similarity_threshold,
        )
        statement = apply_retrieval_filters(statement, filters)
        statement = statement.order_by(desc(similarity), ChunkRecord.chunk_id).limit(config.top_k)
        try:
            result = await session.execute(statement)
            return [
                RetrievalRow(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    score=max(0.0, min(1.0, float(row.similarity))),
                    metadata=row.chunk_metadata,
                    chunk_index=row.chunk_index,
                )
                for row in result
            ]
        except Exception as error:
            raise RetrievalDatabaseError("Vector retrieval query failed.") from error