"""Run the real RAG evaluation through DeepEval's native evaluate API."""

import asyncio
import json
import os
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    PIILeakageMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

from app.core.config import Settings
from app.database.session import create_engine, create_session_factory
from app.evaluation.dataset import load_dataset
from app.evaluation.judge import evaluation_judge_from_settings
from app.embeddings.models import EmbeddingConfig
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.service import EmbeddingService
from app.rag.models import RAGRequest
from app.rag.phase9 import Phase9RAGService
from app.rag.providers.openai import OpenAILLMProvider
from app.rag.query.rewriter import LLMQueryRewriter
from app.rag.relevance.checker import LLMRelevanceChecker
from app.rag.service import RAGService
from app.retrieval.models import HybridConfig, RetrievalConfig
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


DATASET_PATH = Path("evaluation/datasets/sample_rag_dataset.json")
RESULT_PATH = Path("evaluation/results/deepeval_latest.json")


def build_rag_service(settings: Settings) -> Phase9RAGService:
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for the RAG application.")
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
    return Phase9RAGService(
        retrieval_service,
        phase8,
        LLMRelevanceChecker(llm_provider),
        LLMQueryRewriter(llm_provider),
        rewrite_enabled=settings.query_rewrite_enabled,
        relevance_threshold=settings.rag_relevance_threshold,
    )


def build_metrics(settings: Settings):
    judge = evaluation_judge_from_settings(settings)
    common = {"model": judge, "include_reason": True, "async_mode": False}
    rag_threshold = settings.eval_faithfulness_threshold
    safety_threshold = settings.eval_safety_threshold
    return [
        FaithfulnessMetric(threshold=rag_threshold, **common),
        AnswerRelevancyMetric(threshold=settings.eval_answer_relevancy_threshold, **common),
        ContextualPrecisionMetric(threshold=settings.eval_context_precision_threshold, **common),
        ContextualRecallMetric(threshold=settings.eval_context_recall_threshold, **common),
        ContextualRelevancyMetric(threshold=settings.eval_context_relevancy_threshold, **common),
        BiasMetric(threshold=safety_threshold, **common),
        ToxicityMetric(threshold=safety_threshold, **common),
        PIILeakageMetric(threshold=safety_threshold, **common),
    ]


async def collect_test_cases(settings: Settings) -> list[LLMTestCase]:
    samples = load_dataset(DATASET_PATH)
    rag_service = build_rag_service(settings)
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    test_cases: list[LLMTestCase] = []
    try:
        async with session_factory() as session:
            for sample in samples:
                response = await rag_service.answer(
                    session, RAGRequest(query=sample.question)
                )
                test_cases.append(
                    LLMTestCase(
                        name=sample.id,
                        input=sample.question,
                        actual_output=response.answer,
                        expected_output=sample.expected_answer,
                        context=sample.expected_contexts,
                        retrieval_context=response.retrieved_context,
                        metadata=sample.metadata or {},
                    )
                )
    finally:
        await engine.dispose()
    return test_cases


async def main() -> None:
    if os.getenv("RUN_REAL_EVAL") != "1":
        raise SystemExit("Set RUN_REAL_EVAL=1 to run DeepEval evaluation.")
    settings = Settings()
    test_cases = await collect_test_cases(settings)
    result = evaluate(
        test_cases=test_cases,
        metrics=build_metrics(settings),
        identifier="enterprise-rag-deepeval",
        async_config=AsyncConfig(run_async=False, max_concurrent=1, throttle_value=1.0),
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print(f"DeepEval completed for {len(test_cases)} test case(s).")
    print(f"Report: {RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
