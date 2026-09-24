"""Optional offline DeepEval entry point.

This module intentionally requires application wiring and real credentials only when
the optional real-evaluation command is executed.
"""

import asyncio
import os
from pathlib import Path

from app.core.config import Settings
from app.database.session import create_engine, create_session_factory, session_scope
from app.embeddings.models import EmbeddingConfig
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.service import EmbeddingService
from app.evaluation.metrics import build_deepeval_metrics
from app.evaluation.models import evaluation_thresholds_from_settings
from app.evaluation.runner import EvaluationRunner, write_report
from app.rag.phase9 import Phase9RAGService
from app.rag.providers.openai import OpenAILLMProvider
from app.rag.query.rewriter import LLMQueryRewriter
from app.rag.relevance.checker import LLMRelevanceChecker
from app.rag.service import RAGService
from app.retrieval.models import HybridConfig, RetrievalConfig
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


async def main() -> None:
    """Run the configured dataset through the real Phase 9 RAG application."""
    if os.getenv("RUN_REAL_EVAL") != "1":
        raise SystemExit("Set RUN_REAL_EVAL=1 to run optional DeepEval evaluation.")
    settings = Settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for real evaluation.")
    api_key = settings.openai_api_key.get_secret_value()
    embedding_config = EmbeddingConfig(
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    embedding_service = EmbeddingService(
        OpenAIEmbeddingProvider(embedding_config, api_key=api_key), embedding_config
    )
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
    llm_provider = OpenAILLMProvider(
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    phase8 = RAGService(llm_provider=llm_provider, retrieval_service=retrieval_service)
    phase9 = Phase9RAGService(
        retrieval_service,
        phase8,
        LLMRelevanceChecker(llm_provider),
        LLMQueryRewriter(llm_provider),
        rewrite_enabled=settings.query_rewrite_enabled,
        relevance_threshold=settings.rag_relevance_threshold,
    )

    class SessionRAG:
        async def answer(self, session, request):
            async with session_scope(session_factory) as database_session:
                return await phase9.answer(database_session, request)

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    metrics = build_deepeval_metrics(
        evaluation_thresholds_from_settings(settings), settings.eval_llm_model
    )
    try:
        report = await EvaluationRunner(
            SessionRAG(), metrics, settings.eval_llm_model
        ).run(
            Path("evaluation/datasets/sample_rag_dataset.json"),
            dataset_id="sample_rag_dataset",
        )
        write_report(report, "evaluation/results/latest.json")
        print(
            f"Evaluated {report.total_samples} samples; "
            f"passed={report.passed_samples}, failed={report.failed_samples}"
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())