"""Observability isolation tests without LangSmith network access."""

import logging

import pytest

from app.core.config import Settings
from app.observability.service import RAGObservability
from app.rag.models import RAGRequest, RAGResponse
from app.rag.phase9 import Phase9RAGService
from app.rag.relevance.models import RelevanceCheckResult
from app.retrieval.models import HybridRetrievalResult


class FakeSpan:
    def __init__(self, record) -> None:
        self.record = record

    def end(self, *, outputs=None, error=None) -> None:
        self.record["outputs"] = outputs
        self.record["error"] = type(error).__name__ if error else None


class FakeBackend:
    def __init__(self, *, fail_start=False, fail_end=False) -> None:
        self.records = []
        self.fail_start = fail_start
        self.fail_end = fail_end

    def start(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("trace start failed")
        record = dict(kwargs)
        self.records.append(record)
        if self.fail_end:
            class FailingSpan:
                def end(self, **kwargs):
                    raise RuntimeError("trace end failed")
            return FailingSpan()
        return FakeSpan(record)


def candidate() -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="sensitive retrieved chunk content",
        hybrid_score=0.9,
        vector_score=0.9,
        keyword_score=1.0,
        normalized_vector_score=1.0,
        normalized_keyword_score=1.0,
        metadata={"file_type": "pdf"},
        chunk_index=0,
    )


class Retrieval:
    def __init__(self, result):
        self.result = result
        self.queries = []

    async def hybrid_retrieve(self, session, query, filters=None):
        self.queries.append(query)
        return self.result


class Rewriter:
    async def rewrite(self, query):
        return "rewritten query"


class Checker:
    async def check(self, query, candidates):
        return RelevanceCheckResult(relevant=True, confidence=0.9)


class Generator:
    model = "fake-model"
    temperature = 0.0

    async def answer_from_results(self, request, results):
        return RAGResponse(
            answer="same grounded answer",
            sources=[],
            retrieved_context=[result.content for result in results],
        )


@pytest.mark.asyncio
async def test_disabled_observability_runs_operation_without_backend() -> None:
    observability = RAGObservability()
    output = await observability.run(
        name="test",
        stage="rag",
        operation=lambda: _return_value(),
        inputs={"query": "question"},
        output=lambda value, elapsed: {"value": value},
    )

    assert output == "ok"


async def _return_value():
    return "ok"


def test_missing_langsmith_credentials_disable_tracing_without_startup_failure(
    monkeypatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        postgres_password="test-password",
        observability_enabled=True,
        langsmith_tracing=True,
    )

    observability = RAGObservability.from_settings(settings)

    assert observability.enabled is False


def test_observability_defaults_disabled_and_secret_is_masked() -> None:
    settings = Settings(
        _env_file=None,
        postgres_password="test-password",
        langsmith_api_key="ls_secret_value",
        observability_enabled=False,
        langsmith_tracing=False,
    )

    assert settings.observability_enabled is False
    assert "ls_secret_value" not in repr(settings)


@pytest.mark.asyncio
async def test_trace_metadata_is_allowlisted_and_never_contains_api_key(caplog) -> None:
    backend = FakeBackend()
    observability = RAGObservability(enabled=True, backend=backend, environment="test")
    secret = "ls_test_sensitive_token"

    await observability.run(
        name="root",
        stage="rag",
        operation=lambda: _return_value(),
        inputs={"query": "question"},
        output=lambda value, latency: {"answer": value},
    )

    assert backend.records[0]["tags"] == ["rag", "phase-11", "production-rag"]
    assert backend.records[0]["metadata"]["environment"] == "test"
    assert secret not in repr(backend.records)
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [FakeBackend(fail_start=True), FakeBackend(fail_end=True)])
async def test_observability_failures_do_not_change_operation_result(backend) -> None:
    observability = RAGObservability(enabled=True, backend=backend)
    result = await observability.run(
        name="test",
        stage="generation",
        operation=lambda: _return_value(),
        inputs={"context_count": 1},
        output=lambda value, latency: {"value": value},
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_enabled_and_disabled_tracing_preserve_phase9_response() -> None:
    responses = []
    backend = FakeBackend()
    for observer in (RAGObservability(), RAGObservability(enabled=True, backend=backend)):
        retrieval = Retrieval([candidate()])
        service = Phase9RAGService(
            retrieval,
            Generator(),
            Checker(),
            Rewriter(),
            observability=observer,
        )
        response = await service.answer(None, RAGRequest(query="original question"))
        responses.append((response.model_dump(), retrieval.queries))

    assert responses[0] == responses[1]
    assert responses[0][1] == ["rewritten query"]
    names = {record["name"] for record in backend.records}
    assert {
        "RAG request",
        "Query rewrite",
        "Hybrid retrieval",
        "Retrieval relevance check",
        "RAG answer generation",
    } <= names
    assert "sensitive retrieved chunk content" not in repr(backend.records)
