"""Orchestration for the stateless grounded RAG pipeline."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.exceptions import InvalidQueryError
from app.retrieval.models import HybridRetrievalResult, RetrievalFilter
from app.retrieval.service import RetrievalService
from app.rag.context import ContextBuilder
from app.rag.exceptions import EmptyLLMAnswerError, PromptConstructionError
from app.rag.models import RAGRequest, RAGResponse, SourceReference
from app.rag.prompts import PromptBuilder
from app.rag.providers.base import LLMProvider

NO_CONTEXT_ANSWER = "I couldn't find relevant information in the knowledge base to answer this question."


class RAGService:
    """Coordinate hybrid retrieval, bounded context, prompts, and answer generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    async def answer(
        self,
        session: AsyncSession,
        request: RAGRequest,
        filters: RetrievalFilter | None = None,
    ) -> RAGResponse:
        """Return a grounded answer or a deterministic no-context response."""
        if not request.query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        results = await self.retrieval_service.hybrid_retrieve(
            session, request.query, filters
        )
        return await self.answer_from_results(request, results)

    async def answer_from_results(
        self,
        request: RAGRequest,
        results: list[HybridRetrievalResult],
    ) -> RAGResponse:
        """Generate from an already accepted retrieval result set."""
        if not request.query.strip():
            raise InvalidQueryError("Retrieval query must not be empty.")
        if not results:
            return RAGResponse(answer=NO_CONTEXT_ANSWER, sources=[])
        context, context_results = self.context_builder.build_with_sources(results)
        try:
            system_prompt, user_prompt = self.prompt_builder.build(context, request.query)
        except Exception as error:
            if isinstance(error, PromptConstructionError):
                raise
            raise PromptConstructionError("Unable to construct the RAG prompt.") from error
        answer = await self.llm_provider.generate(system_prompt, user_prompt)
        if not answer or not answer.strip():
            raise EmptyLLMAnswerError("The LLM returned an empty answer.")
        return RAGResponse(
            answer=answer.strip(),
            sources=[_source_reference(result) for result in context_results],
        )


def _source_reference(result: HybridRetrievalResult) -> SourceReference:
    """Map an actual retrieval result to a user-visible source reference."""
    return SourceReference(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        metadata=result.metadata,
        chunk_index=result.chunk_index,
        retrieval_score=result.hybrid_score,
    )