"""Embedding generation orchestration."""

from collections.abc import Sequence

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingResponseError,
)
from app.embeddings.models import EmbeddingConfig, EmbeddingResult
from app.ingestion.chunking.models import DocumentChunk


class EmbeddingService:
    """Batch chunks through a provider and validate every returned vector."""

    def __init__(self, provider: EmbeddingProvider, config: EmbeddingConfig) -> None:
        self.provider = provider
        self.config = config

    async def embed_chunks(
        self, chunks: Sequence[DocumentChunk]
    ) -> list[EmbeddingResult]:
        """Return vectors associated with chunks in their original order."""
        if not chunks:
            return []
        if any(not chunk.content.strip() for chunk in chunks):
            raise EmbeddingInputError("Embedding input chunks must contain text.")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise EmbeddingInputError("Embedding input chunk IDs must be unique.")
        results: list[EmbeddingResult] = []
        for offset in range(0, len(chunks), self.config.batch_size):
            batch = list(chunks[offset : offset + self.config.batch_size])
            texts = [chunk.content for chunk in batch]
            try:
                vectors = await self.provider.embed_documents(texts)
            except EmbeddingProviderError:
                raise
            except Exception as error:
                raise EmbeddingProviderError("The embedding provider request failed.") from error
            if len(vectors) != len(batch):
                raise InvalidEmbeddingResponseError(
                    "The embedding provider returned an unexpected vector count."
                )
            for chunk, vector in zip(batch, vectors, strict=True):
                if len(vector) != self.config.dimension:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {self.config.dimension}, "
                        f"received {len(vector)}."
                    )
                results.append(EmbeddingResult(chunk_id=chunk.chunk_id, vector=list(vector)))
        return results

    async def embed_query(self, query: str) -> list[float]:
        """Embed one retrieval query using the same provider contract."""
        if not query.strip():
            raise EmbeddingInputError("Embedding query must not be empty.")
        try:
            vector = await self.provider.embed_text(query)
        except EmbeddingProviderError:
            raise
        except Exception as error:
            raise EmbeddingProviderError("The embedding provider request failed.") from error
        if len(vector) != self.config.dimension:
            raise EmbeddingDimensionError(
                f"Expected embedding dimension {self.config.dimension}, "
                f"received {len(vector)}."
            )
        return list(vector)