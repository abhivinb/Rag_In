"""Errors raised by the RAG answering pipeline."""


class RAGError(Exception):
    """Base class for RAG pipeline failures."""


class LLMProviderError(RAGError):
    """Raised when the language model cannot generate an answer."""


class ContextConstructionError(RAGError):
    """Raised when retrieved chunks cannot form bounded context."""


class PromptConstructionError(RAGError):
    """Raised when grounded prompts cannot be constructed."""


class EmptyLLMAnswerError(LLMProviderError):
    """Raised when the provider returns no usable answer."""


class RelevanceCheckError(RAGError):
    """Raised when relevance checking cannot produce a valid judgment."""