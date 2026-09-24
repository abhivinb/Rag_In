"""Errors raised by vector retrieval."""


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class InvalidQueryError(RetrievalError):
    """Raised when a retrieval query is empty or invalid."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when retrieval configuration is invalid."""


class HybridConfigurationError(RetrievalConfigurationError):
    """Raised when hybrid retrieval configuration is invalid."""


class QueryEmbeddingError(RetrievalError):
    """Raised when the query cannot be embedded."""


class RetrievalDatabaseError(RetrievalError):
    """Raised when vector retrieval cannot query PostgreSQL."""