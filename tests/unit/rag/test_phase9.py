"""Phase 9 orchestration tests with deterministic fakes."""

import pytest

from app.rag.models import RAGRequest, RAGResponse
from app.rag.phase9 import INSUFFICIENT_CONTEXT_ANSWER, Phase9RAGService
from app.rag.relevance.models import RelevanceCheckResult
from app.retrieval.models import HybridRetrievalResult


def result(chunk_id: str) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"content for {chunk_id}",
        hybrid_score=0.8,
        vector_score=0.8,
        keyword_score=0.5,
        normalized_vector_score=1.0,
        normalized_keyword_score=0.5,
        metadata={"file_name": "guide.pdf"},
        chunk_index=0,
    )


class FakeRetrieval:
    def __init__(self, result_sets):
        self.result_sets = list(result_sets)
        self.queries: list[str] = []

    async def hybrid_retrieve(self, session, query, filters=None):
        self.queries.append(query)
        return self.result_sets.pop(0) if self.result_sets else []


class FakeRewriter:
    def __init__(self, rewritten: str = "rewritten query", fail: bool = False):
        self.rewritten = rewritten
        self.fail = fail
        self.calls = 0

    async def rewrite(self, query: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("rewrite failed")
        return self.rewritten


class FakeChecker:
    def __init__(self, checks):
        self.checks = list(checks)
        self.calls: list[str] = []

    async def check(self, query, candidates):
        self.calls.append(query)
        return self.checks.pop(0)


class FakeRAG:
    def __init__(self):
        self.calls = []

    async def answer_from_results(self, request, results):
        self.calls.append(results)
        return RAGResponse(answer="accepted", sources=[])


def check(relevant: bool, confidence: float) -> RelevanceCheckResult:
    return RelevanceCheckResult(
        relevant=relevant,
        confidence=confidence,
        reason="test",
        relevant_chunk_ids=[],
    )


def service(retrieval, checker, rag, rewriter=None, **kwargs):
    return Phase9RAGService(
        retrieval,
        rag,
        checker,
        rewriter,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_rewrite_then_relevance_passes_to_phase8() -> None:
    retrieval = FakeRetrieval([[result("rewritten")]])
    checker = FakeChecker([check(True, 0.9)])
    rag = FakeRAG()
    rewriter = FakeRewriter()

    response = await service(retrieval, checker, rag, rewriter).answer(
        None, RAGRequest(query="unclear question")
    )

    assert response.answer == "accepted"
    assert retrieval.queries == ["rewritten query"]
    assert checker.calls == ["rewritten query"]
    assert len(rag.calls) == 1


@pytest.mark.asyncio
async def test_rewrite_failure_falls_back_to_original_query() -> None:
    retrieval = FakeRetrieval([[result("original")]])
    checker = FakeChecker([check(True, 0.9)])

    await service(retrieval, checker, FakeRAG(), FakeRewriter(fail=True)).answer(
        None, RAGRequest(query="original query")
    )

    assert retrieval.queries == ["original query"]


@pytest.mark.asyncio
async def test_rewrite_disabled_skips_rewriter() -> None:
    retrieval = FakeRetrieval([[result("original")]])
    rewriter = FakeRewriter()

    await service(
        retrieval,
        FakeChecker([check(True, 0.9)]),
        FakeRAG(),
        rewriter,
        rewrite_enabled=False,
    ).answer(None, RAGRequest(query="explicit query"))

    assert rewriter.calls == 0
    assert retrieval.queries == ["explicit query"]


@pytest.mark.asyncio
async def test_no_results_skip_relevance_and_generation() -> None:
    retrieval = FakeRetrieval([[]])
    checker = FakeChecker([])
    rag = FakeRAG()

    response = await service(retrieval, checker, rag).answer(
        None, RAGRequest(query="missing")
    )

    assert response.sources == []
    assert checker.calls == []
    assert rag.calls == []


@pytest.mark.asyncio
async def test_irrelevant_first_attempt_retries_once_and_accepts_retry_sources() -> None:
    retrieval = FakeRetrieval([[result("rejected")], [result("accepted")]])
    checker = FakeChecker([check(False, 0.9), check(True, 0.8)])
    rag = FakeRAG()

    response = await service(
        retrieval, checker, rag, FakeRewriter("rewritten")
    ).answer(None, RAGRequest(query="original"))

    assert retrieval.queries == ["rewritten", "original"]
    assert checker.calls == ["rewritten", "original"]
    assert rag.calls[0][0].chunk_id == "accepted"
    assert response.answer == "accepted"


@pytest.mark.asyncio
async def test_irrelevant_retry_returns_safe_no_answer_and_never_generates() -> None:
    retrieval = FakeRetrieval([[result("first")], [result("second")]])
    checker = FakeChecker([check(False, 0.9), check(True, 0.2)])
    rag = FakeRAG()

    response = await service(retrieval, checker, rag).answer(
        None, RAGRequest(query="question")
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == []
    assert rag.calls == []
    assert len(retrieval.queries) == 2


@pytest.mark.asyncio
async def test_relevant_false_even_with_high_confidence_is_rejected() -> None:
    retrieval = FakeRetrieval([[result("first")], []])
    checker = FakeChecker([check(False, 1.0)])

    response = await service(retrieval, checker, FakeRAG()).answer(
        None, RAGRequest(query="question")
    )

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER