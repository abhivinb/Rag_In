"""Embedding service tests using a fake provider."""

import pytest

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import (
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingResponseError,
)
from app.embeddings.models import EmbeddingConfig
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking.models import ChunkMetadata, DocumentChunk


class FakeProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index)] * self.dimension for index, _ in enumerate(texts)]


class FailingProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingProviderError("provider failed")


class WrongCountProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []


class CountingProvider(FakeProvider):
    pass


def make_chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-id",
        content=content,
        chunk_index=0,
        metadata=ChunkMetadata(
            source="example.txt",
            file_name="example.txt",
            file_type="txt",
            source_type="file",
            chunk_start=0,
            chunk_end=len(content),
        ),
    )


@pytest.mark.asyncio
async def test_embedding_service_batches_chunks_and_preserves_ids() -> None:
    provider = FakeProvider()
    service = EmbeddingService(
        provider, EmbeddingConfig(dimension=1536, batch_size=2)
    )
    chunks = [make_chunk(f"chunk-{index}", f"text-{index}") for index in range(5)]

    results = await service.embed_chunks(chunks)

    assert [result.chunk_id for result in results] == [f"chunk-{i}" for i in range(5)]
    assert provider.calls == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]
    assert all(len(result.vector) == 1536 for result in results)


@pytest.mark.asyncio
async def test_empty_input_returns_no_vectors() -> None:
    service = EmbeddingService(FakeProvider(), EmbeddingConfig(dimension=1536))

    assert await service.embed_chunks([]) == []


def test_unsupported_model_dimension_is_rejected_before_provider_use() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        EmbeddingConfig(model="unsupported-model", dimension=1536)
    with pytest.raises(ValueError, match="requires dimension 1536"):
        EmbeddingConfig(model="text-embedding-3-small", dimension=3072)


def test_invalid_configuration_makes_no_provider_call() -> None:
    provider = CountingProvider()

    with pytest.raises(ValueError):
        EmbeddingConfig(dimension=3072)
    assert provider.calls == []


@pytest.mark.asyncio
async def test_non_transient_openai_error_is_not_retried() -> None:
    class Embeddings:
        calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise ValueError("invalid request")

    class Client:
        def __init__(self) -> None:
            self.embeddings = Embeddings()

    client = Client()
    provider = OpenAIEmbeddingProvider(
        EmbeddingConfig(dimension=1536, model="text-embedding-3-small"),
        api_key="test-key",
        client=client,
    )

    with pytest.raises(EmbeddingProviderError):
        await provider.embed_documents(["text"])
    assert client.embeddings.calls == 1


@pytest.mark.asyncio
async def test_provider_failure_is_propagated() -> None:
    service = EmbeddingService(FailingProvider(), EmbeddingConfig(dimension=1536))

    with pytest.raises(EmbeddingProviderError):
        await service.embed_chunks([make_chunk("chunk", "text")])


@pytest.mark.asyncio
async def test_vector_count_is_validated() -> None:
    service = EmbeddingService(WrongCountProvider(), EmbeddingConfig(dimension=1536))

    with pytest.raises(InvalidEmbeddingResponseError):
        await service.embed_chunks([make_chunk("chunk", "text")])


@pytest.mark.asyncio
async def test_vector_dimension_is_validated() -> None:
    service = EmbeddingService(
        FakeProvider(dimension=2), EmbeddingConfig(dimension=1536)
    )

    with pytest.raises(EmbeddingDimensionError):
        await service.embed_chunks([make_chunk("chunk", "text")])


@pytest.mark.asyncio
async def test_whitespace_and_duplicate_inputs_are_rejected() -> None:
    service = EmbeddingService(FakeProvider(), EmbeddingConfig(dimension=1536))

    with pytest.raises(EmbeddingInputError):
        await service.embed_chunks([make_chunk("chunk", " ")])
    with pytest.raises(EmbeddingInputError):
        await service.embed_chunks([make_chunk("chunk", "one"), make_chunk("chunk", "two")])