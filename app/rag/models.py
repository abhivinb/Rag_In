"""Pydantic contracts for stateless RAG requests and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RAGRequest(BaseModel):
    """One user question for the RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: str


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