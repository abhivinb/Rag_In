"""Pydantic contracts for stateless RAG requests and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGRequest(BaseModel):
    """One user question for the RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject invisible control characters before they reach prompts/logs."""
        normalized = value.strip()
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError("query contains unsupported control characters")
        return normalized


class SourceReference(BaseModel):
    """A source derived directly from a retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    metadata: dict[str, Any]
    chunk_index: int = Field(ge=0)
    retrieval_score: float = Field(ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    """Grounded answer and deterministic source references."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sources: list[SourceReference]
    retrieved_context: list[str] = Field(default_factory=list)
