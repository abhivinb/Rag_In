"""Provider abstraction for embedding generation."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Provider-independent interface used by the embedding service."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate one vector per input text in the same order."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate one vector for a single text."""
        vectors = await self.embed_documents([text])
        return vectors[0]