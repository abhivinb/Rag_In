"""Application dependencies and production RAG service wiring."""

from fastapi import HTTPException, Request, status

from app.conversation.repository import ConversationRepository
from app.core.config import Settings, get_settings
from app.embeddings.models import EmbeddingConfig
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.service import EmbeddingService
from app.database.persistence import EmbeddingPersistenceService
from app.database.repository import VectorRepository
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.indexing import DocumentIndexingService
from app.ingestion.service import IngestionService
from app.observability import RAGObservability
from app.rag.phase9 import Phase9RAGService
from app.rag.providers.openai import OpenAILLMProvider
from app.rag.query.rewriter import LLMQueryRewriter
from app.rag.relevance.checker import LLMRelevanceChecker
from app.rag.context import ContextBuilder
from app.rag.service import RAGService
from app.retrieval.models import HybridConfig, RetrievalConfig
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


def build_rag_service(settings: Settings) -> Phase9RAGService:
    """Build the real Phase 9 pipeline from environment-backed settings."""
    if settings.llm_provider.lower() != "openai":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise ValueError("OPENAI_API_KEY is required for the RAG endpoint.")

    embedding_config = EmbeddingConfig(
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    api_key = settings.openai_api_key.get_secret_value()
    llm_provider = OpenAILLMProvider(
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    embedding_service = _build_embedding_service(settings, embedding_config, api_key)
    retrieval_service = RetrievalService(
        embedding_service,
        VectorRetrievalRepository(),
        RetrievalConfig(
            top_k=settings.retrieval_top_k,
            similarity_threshold=settings.retrieval_similarity_threshold,
        ),
        HybridConfig(
            vector_weight=settings.hybrid_vector_weight,
            keyword_weight=settings.hybrid_keyword_weight,
            candidate_multiplier=settings.hybrid_candidate_multiplier,
        ),
    )
    return Phase9RAGService(
        retrieval_service,
        RAGService(
            retrieval_service,
            llm_provider,
            context_builder=ContextBuilder(settings.rag_max_context_chars),
        ),
        LLMRelevanceChecker(llm_provider),
        LLMQueryRewriter(llm_provider),
        rewrite_enabled=settings.query_rewrite_enabled,
        relevance_threshold=settings.rag_relevance_threshold,
        observability=RAGObservability.from_settings(settings),
    )


def build_document_indexing_service(settings: Settings) -> DocumentIndexingService:
    """Build the ingestion workflow with the configured embedding provider."""
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise ValueError("OPENAI_API_KEY is required for document indexing.")
    embedding_config = EmbeddingConfig(
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    embedding_service = _build_embedding_service(
        settings, embedding_config, settings.openai_api_key.get_secret_value()
    )
    return DocumentIndexingService(
        IngestionService(),
        ChunkingService(),
        EmbeddingPersistenceService(embedding_service, VectorRepository()),
    )


def _build_embedding_service(
    settings: Settings, config: EmbeddingConfig, api_key: str
) -> EmbeddingService:
    """Create the shared OpenAI embedding service."""
    return EmbeddingService(
        OpenAIEmbeddingProvider(config, api_key=api_key),
        config,
    )


def get_rag_service(request: Request) -> Phase9RAGService:
    """Return one cached service graph for the application process."""
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        try:
            service = build_rag_service(get_settings())
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        request.app.state.rag_service = service
    return service


def get_document_indexing_service(request: Request) -> DocumentIndexingService:
    """Return one cached document indexing service for the process."""
    service = getattr(request.app.state, "document_indexing_service", None)
    if service is None:
        try:
            service = build_document_indexing_service(get_settings())
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        request.app.state.document_indexing_service = service
    return service


async def get_database_session(request: Request):
    """Yield one short-lived database session per request."""
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")
    async with session_factory() as session:
        yield session


def get_conversation_repository() -> ConversationRepository:
    """Return the stateless repository used when memory is enabled."""
    return ConversationRepository()
