"""Retrieval service tests without OpenAI or PostgreSQL."""

from collections.abc import Sequence

import pytest

from app.embeddings.base import EmbeddingProvider
from app.embeddings.models import EmbeddingConfig
from app.embeddings.service import EmbeddingService
from app.retrieval.exceptions import InvalidQueryError, QueryEmbeddingError, RetrievalConfigurationError
from app.retrieval.models import RetrievalConfig, RetrievalFilter
from app.retrieval.repository import RetrievalRow
from app.retrieval.service import RetrievalService


class FakeProvider(EmbeddingProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("provider failure")
        self.queries.extend(texts)
        return [[1.0] + [0.0] * 1535 for _ in texts]


class FakeRepository:
    def __init__(self, rows: Sequence[RetrievalRow] = ()) -> None:
        self.rows = list(rows)
        self.query_vectors: list[list[float]] = []
        self.filters: list[RetrievalFilter | None] = []

    async def search(self, session, query_vector, config, filters=None):
        self.query_vectors.append(list(query_vector))
        self.filters.append(filters)
        return self.rows[: config.top_k]


def make_service(*, rows=(), fail=False, config=None):
    provider = FakeProvider(fail=fail)
    embedding = EmbeddingService(provider, EmbeddingConfig())
    repository = FakeRepository(rows)
    service = RetrievalService(embedding, repository, config or RetrievalConfig())
    return service, repository, provider


def make_row(score: float = 0.8) -> RetrievalRow:
    return RetrievalRow(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="content",
        score=score,
        metadata={"file_type": "pdf"},
        chunk_index=0,
    )


@pytest.mark.asyncio
async def test_service_embeds_query_and_converts_results() -> None:
    service, repository, provider = make_service(rows=[make_row()])

    results = await service.retrieve(None, "what is this?", RetrievalFilter(file_type="pdf"))

    assert len(results) == 1
    assert results[0].score == 0.8
    assert provider.queries == ["what is this?"]
    assert len(repository.query_vectors[0]) == 1536
    assert repository.filters[0].file_type == "pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "  \n\t"])
async def test_empty_query_is_rejected(query: str) -> None:
    service, _, _ = make_service()

    with pytest.raises(InvalidQueryError):
        await service.retrieve(None, query)


@pytest.mark.asyncio
async def test_query_embedding_failure_is_wrapped() -> None:
    service, _, _ = make_service(fail=True)

    with pytest.raises(QueryEmbeddingError):
        await service.retrieve(None, "query")


def test_service_rejects_invalid_configuration() -> None:
    invalid_config = RetrievalConfig.model_construct(top_k=0, similarity_threshold=0.0)

    with pytest.raises(RetrievalConfigurationError):
        RetrievalService(
            EmbeddingService(FakeProvider(), EmbeddingConfig()),
            FakeRepository(),
            invalid_config,
        )