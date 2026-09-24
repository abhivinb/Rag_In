"""Query embedding and vector retrieval orchestration."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.exceptions import EmbeddingError
from app.embeddings.service import EmbeddingService
from app.retrieval.exceptions import (
    HybridConfigurationError,
    InvalidQueryError,
    QueryEmbeddingError,
    RetrievalConfigurationError,
)
from app.retrieval.models import (
    HybridConfig,
    HybridRetrievalResult,
    RetrievalConfig,
    RetrievalFilter,
    RetrievalResult,
    validate_retrieval_config,
)
from app.retrieval.repository import RetrievalRow, VectorRetrievalRepository
from app.retrieval.fusion import fuse_rows


class RetrievalService:
    """Embed a query and return ranked results from pgvector."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        repository: VectorRetrievalRepository,
        config: RetrievalConfig,
        hybrid_config: HybridConfig | None = None,
    ) -> None:
        try:
            validate_retrieval_config(config)
        except ValueError as error:
            raise RetrievalConfigurationError(str(error)) from error
        self.embedding_service = embedding_service
        self.repository = repository
        self.config = config
        self.hybrid_config = hybrid_config or HybridConfig()
        try:
            self.hybrid_config.validate_weights()
        except ValueError as error:
            raise HybridConfigurationError(str(error)) from error

    async def hybrid_retrieve(
        self,
        session: AsyncSession,
        query: str,
        filters: RetrievalFilter | None = None,
        hybrid_config: HybridConfig | None = None,
    ) -> list[HybridRetrievalResult]:
        """Fuse vector and PostgreSQL full-text candidates into ranked results."""
        if not query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        config = hybrid_config or self.hybrid_config
        try:
            config.validate_weights()
        except ValueError as error:
            raise HybridConfigurationError(str(error)) from error
        candidate_limit = self.config.top_k * config.candidate_multiplier
        try:
            query_vector = await self.embedding_service.embed_query(query)
        except EmbeddingError as error:
            raise QueryEmbeddingError("Unable to embed the retrieval query.") from error
        vector_config = self.config.model_copy(update={"top_k": candidate_limit})
        vector_rows, keyword_rows = await _retrieve_candidates(
            self.repository,
            session,
            query,
            query_vector,
            vector_config,
            filters,
            candidate_limit,
        )
        fused = fuse_rows(vector_rows, keyword_rows, config)
        return [_to_hybrid_result(item) for item in fused[: self.config.top_k]]

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


async def _retrieve_candidates(
    repository: VectorRetrievalRepository,
    session: AsyncSession,
    query: str,
    query_vector: Sequence[float],
    vector_config: RetrievalConfig,
    filters: RetrievalFilter | None,
    candidate_limit: int,
) -> tuple[list[RetrievalRow], list[RetrievalRow]]:
    """Run both candidate queries with the same filters."""
    vector_rows = await repository.search(session, query_vector, vector_config, filters)
    keyword_rows = await repository.keyword_search(
        session, query, candidate_limit, filters
    )
    return vector_rows, keyword_rows


def _to_hybrid_result(row) -> HybridRetrievalResult:
    """Convert a fused projection to the public hybrid result model."""
    return HybridRetrievalResult(
        chunk_id=row.row.chunk_id,
        document_id=row.row.document_id,
        content=row.row.content,
        hybrid_score=row.hybrid_score,
        vector_score=row.vector_score,
        keyword_score=row.keyword_score,
        normalized_vector_score=row.normalized_vector_score,
        normalized_keyword_score=row.normalized_keyword_score,
        metadata=row.row.metadata,
        chunk_index=row.row.chunk_index,
    )