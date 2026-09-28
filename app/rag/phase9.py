"""Phase 9 bounded query rewrite and relevance-aware RAG orchestration."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.exceptions import RelevanceCheckError
from app.rag.models import RAGRequest, RAGResponse
from app.observability.service import RAGObservability
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
        observability: RAGObservability | None = None,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0.0 and 1.0.")
        self.retrieval_service = retrieval_service
        self.rag_service = rag_service
        self.relevance_checker = relevance_checker
        self.query_rewriter = query_rewriter
        self.rewrite_enabled = rewrite_enabled
        self.relevance_threshold = relevance_threshold
        self.observability = observability or RAGObservability()

    async def answer(
        self,
        session: AsyncSession,
        request: RAGRequest,
        filters: RetrievalFilter | None = None,
    ) -> RAGResponse:
        """Execute exactly one initial attempt and at most one retry."""
        state: dict[str, object] = {
            "rewritten_query": request.query,
            "retrieval_attempts": 0,
        }
        return await self.observability.run(
            name="RAG request",
            stage="rag",
            run_type="chain",
            inputs={"query": request.query},
            operation=lambda: self._answer(session, request, filters, state),
            output=lambda response, latency: {
                "final_query": state["rewritten_query"],
                "retrieval_attempt_count": state["retrieval_attempts"],
                "answer": response.answer,
                "source_count": len(response.sources),
                "success": True,
                "latency_ms": latency,
            },
        )

    async def _answer(self, session, request, filters, state) -> RAGResponse:
        if not request.query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        retrieval_query = request.query
        if self.rewrite_enabled and self.query_rewriter is not None:
            try:
                rewritten = await self.observability.run(
                    name="Query rewrite",
                    stage="query-rewrite",
                    run_type="chain",
                    inputs={"original_query": request.query, "enabled": True},
                    operation=lambda: self.query_rewriter.rewrite(request.query),
                    output=lambda value, latency: {
                        "rewritten_query": value,
                        "fallback": False,
                        "latency_ms": latency,
                    },
                )
                if rewritten.strip():
                    retrieval_query = rewritten.strip()
            except Exception:
                logger.warning("Query rewrite failed; using the original query")
                retrieval_query = request.query
                await self.observability.run(
                    name="Query rewrite fallback",
                    stage="query-rewrite",
                    run_type="chain",
                    inputs={"original_query": request.query, "fallback": True},
                    operation=lambda: _resolved(request.query),
                    output=lambda value, latency: {
                        "retrieval_query": value,
                        "fallback": True,
                        "latency_ms": latency,
                    },
                )
            state["rewritten_query"] = retrieval_query
        elif not self.rewrite_enabled:
            await self.observability.run(
                name="Query rewrite skipped",
                stage="query-rewrite",
                run_type="chain",
                inputs={"original_query": request.query, "enabled": False},
                operation=lambda: _resolved(request.query),
                output=lambda value, latency: {
                    "retrieval_query": value,
                    "skipped": True,
                    "latency_ms": latency,
                },
            )

        first_results = await self._retrieve(
            session, retrieval_query, filters, attempt=1, state=state
        )
        if not first_results:
            return RAGResponse(answer=NO_CONTEXT_ANSWER, sources=[])
        first_check = await self._check(retrieval_query, first_results, attempt=1)
        if self._accepted(first_check):
            return await self._generate(request, first_results)

        retry_query = request.query if retrieval_query != request.query else retrieval_query
        state["rewritten_query"] = retry_query
        retry_results = await self._retrieve(
            session, retry_query, filters, attempt=2, state=state
        )
        if not retry_results:
            return RAGResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])
        retry_check = await self._check(retry_query, retry_results, attempt=2)
        if not self._accepted(retry_check):
            return RAGResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])
        return await self._generate(request, retry_results)

    async def _check(
        self, query: str, results: list[HybridRetrievalResult], *, attempt: int
    ) -> RelevanceCheckResult:
        try:
            return await self.observability.run(
                name="Retrieval relevance check",
                stage="relevance",
                run_type="chain",
                inputs={
                    "query": query,
                    "retrieval_attempt": attempt,
                    "candidate_count": len(results),
                    "chunk_ids": [item.chunk_id for item in results],
                    "threshold": self.relevance_threshold,
                },
                operation=lambda: self.relevance_checker.check(query, results),
                output=lambda result, latency: {
                    "relevant": result.relevant,
                    "confidence": result.confidence,
                    "retry_triggered": not self._accepted(result) and attempt == 1,
                    "latency_ms": latency,
                },
            )
        except Exception as error:
            if isinstance(error, RelevanceCheckError):
                raise
            raise RelevanceCheckError("Retrieval relevance checking failed.") from error

    async def _retrieve(self, session, query, filters, *, attempt, state):
        state["retrieval_attempts"] = attempt
        filter_values = filters.model_dump(exclude_none=True) if filters else {}
        retrieval_config = getattr(self.retrieval_service, "config", None)
        top_k = getattr(retrieval_config, "top_k", None)
        return await self.observability.run(
            name="Hybrid retrieval",
            stage="retrieval",
            run_type="retriever",
            inputs={
                "query": query,
                "top_k": top_k,
                "filters": filter_values,
                "retrieval_attempt": attempt,
            },
            operation=lambda: self.retrieval_service.hybrid_retrieve(
                session, query, filters
            ),
            output=lambda results, latency: {
                "fused_result_count": len(results),
                "latency_ms": latency,
            },
        )

    async def _generate(self, request, results):
        llm = getattr(self.rag_service, "llm_provider", None)
        return await self.observability.run(
            name="RAG answer generation",
            stage="generation",
            run_type="chain",
            inputs={
                "query": request.query,
                "context_chunk_count": len(results),
                "context_chars": sum(len(item.content) for item in results),
                "model": getattr(llm, "model", None),
                "temperature": getattr(llm, "temperature", None),
            },
            operation=lambda: self.rag_service.answer_from_results(request, results),
            output=lambda response, latency: {
                "answer": response.answer,
                "source_count": len(response.sources),
                "latency_ms": latency,
            },
        )

    def _accepted(self, result: RelevanceCheckResult) -> bool:
        return result.relevant and result.confidence >= self.relevance_threshold


async def _resolved(value):
    return value