"""Query embedding and vector retrieval orchestration."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.exceptions import EmbeddingError
from app.embeddings.service import EmbeddingService
from app.retrieval.exceptions import (
    InvalidQueryError,
    QueryEmbeddingError,
    RetrievalConfigurationError,
)
from app.retrieval.models import (
    RetrievalConfig,
    RetrievalFilter,
    RetrievalResult,
    validate_retrieval_config,
)
from app.retrieval.repository import RetrievalRow, VectorRetrievalRepository


class RetrievalService:
    """Embed a query and return ranked results from pgvector."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        repository: VectorRetrievalRepository,
        config: RetrievalConfig,
    ) -> None:
        try:
            validate_retrieval_config(config)
        except ValueError as error:
            raise RetrievalConfigurationError(str(error)) from error
        self.embedding_service = embedding_service
        self.repository = repository
        self.config = config

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievalResult]:
        """Return zero or more ranked results for a user query."""
        if not query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        try:
            query_vector = await self.embedding_service.embed_query(query)
        except EmbeddingError as error:
            raise QueryEmbeddingError("Unable to embed the retrieval query.") from error
        rows = await self.repository.search(session, query_vector, self.config, filters)
        return [_to_result(row) for row in rows]


def _to_result(row: RetrievalRow) -> RetrievalResult:
    """Convert the repository projection to the public retrieval model."""
    return RetrievalResult(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        content=row.content,
        score=row.score,
        metadata=row.metadata,
        chunk_index=row.chunk_index,
    )