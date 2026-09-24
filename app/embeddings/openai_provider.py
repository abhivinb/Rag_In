"""OpenAI embedding provider implementation."""

import asyncio
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import EmbeddingProviderError
from app.embeddings.models import EmbeddingConfig

logger = logging.getLogger(__name__)


def _is_transient(error: Exception) -> bool:
    """Return whether an OpenAI failure is appropriate for bounded retry."""
    return isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    ) or isinstance(error, APIStatusError) and error.status_code >= 500


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings through the official asynchronous OpenAI SDK."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        api_key: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required for the OpenAI provider.")
        self.config = config
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate vectors with bounded retries and no sensitive logging."""
        if not texts:
            return []
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    dimensions=self.config.dimension,
                )
                return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            except Exception as error:
                last_error = error
                if _is_transient(error) and attempt < self.config.max_retries:
                    await asyncio.sleep(2**attempt)
                else:
                    raise EmbeddingProviderError(
                        "The embedding provider request failed."
                    ) from error
        logger.exception("OpenAI embedding request failed after retries")
        raise EmbeddingProviderError("The embedding provider request failed.") from last_error