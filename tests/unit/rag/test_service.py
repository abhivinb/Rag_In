"""RAG service tests with fake retrieval and LLM providers."""

import pytest

from app.rag.exceptions import EmptyLLMAnswerError, LLMProviderError
from app.rag.models import RAGRequest
from app.rag.service import NO_CONTEXT_ANSWER, RAGService
from app.retrieval.models import HybridRetrievalResult


def result(chunk_id: str = "chunk-1") -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="The answer is in this source.",
        hybrid_score=0.9,
        vector_score=0.9,
        keyword_score=1.0,
        normalized_vector_score=1.0,
        normalized_keyword_score=1.0,
        metadata={"file_name": "guide.pdf"},
        chunk_index=2,
    )


class FakeRetrieval:
    def __init__(self, results):
        self.results = results
        self.queries: list[str] = []

    async def hybrid_retrieve(self, session, query, filters=None):
        self.queries.append(query)
        return self.results


class FakeLLM:
    def __init__(self, answer: str = "Grounded answer."):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


@pytest.mark.asyncio
async def test_rag_service_generates_answer_and_sources() -> None:
    retrieval = FakeRetrieval([result()])
    llm = FakeLLM()
    service = RAGService(retrieval, llm)

    response = await service.answer(None, RAGRequest(query="What is the answer?"))

    assert response.answer == "Grounded answer."
    assert response.sources[0].chunk_id == "chunk-1"
    assert "The answer is in this source." in llm.calls[0][1]
    assert retrieval.queries == ["What is the answer?"]


@pytest.mark.asyncio
async def test_no_results_do_not_call_llm() -> None:
    retrieval = FakeRetrieval([])
    llm = FakeLLM()

    response = await RAGService(retrieval, llm).answer(None, RAGRequest(query="Unknown"))

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_empty_llm_answer_is_rejected() -> None:
    with pytest.raises(EmptyLLMAnswerError):
        await RAGService(FakeRetrieval([result()]), FakeLLM("  ")).answer(
            None, RAGRequest(query="Question")
        )


@pytest.mark.asyncio
async def test_llm_failure_is_not_fabricated() -> None:
    class FailingLLM(FakeLLM):
        async def generate(self, system_prompt: str, user_prompt: str) -> str:
            raise LLMProviderError("failed")

    with pytest.raises(LLMProviderError):
        await RAGService(FakeRetrieval([result()]), FailingLLM()).answer(
            None, RAGRequest(query="Question")
        )