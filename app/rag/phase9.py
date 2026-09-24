"""Phase 9 bounded query rewrite and relevance-aware RAG orchestration."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.exceptions import RelevanceCheckError
from app.rag.models import RAGRequest, RAGResponse
from app.rag.query.rewriter import QueryRewriter
from app.rag.relevance.checker import RelevanceChecker
from app.rag.relevance.models import RelevanceCheckResult
from app.rag.service import NO_CONTEXT_ANSWER, RAGService
from app.retrieval.exceptions import InvalidQueryError
from app.retrieval.models import HybridRetrievalResult, RetrievalFilter
from app.retrieval.service import RetrievalService

MAX_RETRIEVAL_ATTEMPTS = 2
logger = logging.getLogger(__name__)
INSUFFICIENT_CONTEXT_ANSWER = (
    "I couldn't find sufficient information in the knowledge base to answer this question."
)


class Phase9RAGService:
    """Run rewrite, at most two retrieval checks, and Phase 8 generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        rag_service: RAGService,
        relevance_checker: RelevanceChecker,
        query_rewriter: QueryRewriter | None = None,
        *,
        rewrite_enabled: bool = True,
        relevance_threshold: float = 0.70,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0.0 and 1.0.")
        self.retrieval_service = retrieval_service
        self.rag_service = rag_service
        self.relevance_checker = relevance_checker
        self.query_rewriter = query_rewriter
        self.rewrite_enabled = rewrite_enabled
        self.relevance_threshold = relevance_threshold

    async def answer(
        self,
        session: AsyncSession,
        request: RAGRequest,
        filters: RetrievalFilter | None = None,
    ) -> RAGResponse:
        """Execute exactly one initial attempt and at most one retry."""
        if not request.query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        retrieval_query = request.query
        if self.rewrite_enabled and self.query_rewriter is not None:
            try:
                rewritten = await self.query_rewriter.rewrite(request.query)
                if rewritten.strip():
                    retrieval_query = rewritten.strip()
            except Exception:
                logger.warning("Query rewrite failed; using the original query")
                retrieval_query = request.query

        first_results = await self.retrieval_service.hybrid_retrieve(
            session, retrieval_query, filters
        )
        if not first_results:
            return RAGResponse(answer=NO_CONTEXT_ANSWER, sources=[])
        first_check = await self._check(retrieval_query, first_results)
        if self._accepted(first_check):
            return await self.rag_service.answer_from_results(request, first_results)

        retry_query = request.query if retrieval_query != request.query else retrieval_query
        retry_results = await self.retrieval_service.hybrid_retrieve(
            session, retry_query, filters
        )
        if not retry_results:
            return RAGResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])
        retry_check = await self._check(retry_query, retry_results)
        if not self._accepted(retry_check):
            return RAGResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])
        return await self.rag_service.answer_from_results(request, retry_results)

    async def _check(
        self, query: str, results: list[HybridRetrievalResult]
    ) -> RelevanceCheckResult:
        try:
            return await self.relevance_checker.check(query, results)
        except Exception as error:
            if isinstance(error, RelevanceCheckError):
                raise
            raise RelevanceCheckError("Retrieval relevance checking failed.") from error

    def _accepted(self, result: RelevanceCheckResult) -> bool:
        return result.relevant and result.confidence >= self.relevance_threshold