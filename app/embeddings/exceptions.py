"""Errors raised during embedding generation."""


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the provider cannot generate embeddings."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when a vector does not match the configured dimension."""


class InvalidEmbeddingResponseError(EmbeddingError):
    """Raised when a provider returns an invalid vector response."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding service configuration is invalid."""


class EmbeddingInputError(EmbeddingError):
    """Raised when chunks cannot be sent for embedding."""