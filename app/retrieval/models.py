"""Models for vector retrieval configuration, filters, and results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RETRIEVAL_DEFAULT_TOP_K = 5
RETRIEVAL_DEFAULT_SIMILARITY_THRESHOLD = 0.0
HYBRID_DEFAULT_VECTOR_WEIGHT = 0.7
HYBRID_DEFAULT_KEYWORD_WEIGHT = 0.3
HYBRID_DEFAULT_CANDIDATE_MULTIPLIER = 3


class RetrievalConfig(BaseModel):
    """Database-side retrieval controls."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=RETRIEVAL_DEFAULT_TOP_K, gt=0)
    similarity_threshold: float = Field(
        default=RETRIEVAL_DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )


class RetrievalFilter(BaseModel):
    """Explicit safe filters supported by the retrieval repository."""

    model_config = ConfigDict(extra="forbid")

    document_id: str | None = None
    source_type: str | None = None
    file_type: str | None = None


class RetrievalResult(BaseModel):
    """Database-independent ranked retrieval result."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any]
    chunk_index: int = Field(ge=0)


class HybridConfig(BaseModel):
    """Configuration for vector and PostgreSQL keyword score fusion."""

    model_config = ConfigDict(extra="forbid")

    vector_weight: float = Field(default=HYBRID_DEFAULT_VECTOR_WEIGHT, ge=0.0)
    keyword_weight: float = Field(default=HYBRID_DEFAULT_KEYWORD_WEIGHT, ge=0.0)
    candidate_multiplier: int = Field(default=HYBRID_DEFAULT_CANDIDATE_MULTIPLIER, ge=1)

    def validate_weights(self) -> None:
        if abs((self.vector_weight + self.keyword_weight) - 1.0) > 1e-9:
            raise ValueError("vector_weight and keyword_weight must sum to 1.0.")


class HybridRetrievalResult(BaseModel):
    """Ranked result containing both source scores and the fused score."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    content: str
    hybrid_score: float = Field(ge=0.0, le=1.0)
    vector_score: float = Field(ge=0.0, le=1.0)
    keyword_score: float = Field(ge=0.0)
    normalized_vector_score: float = Field(ge=0.0, le=1.0)
    normalized_keyword_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any]
    chunk_index: int = Field(ge=0)


def validate_retrieval_config(config: RetrievalConfig) -> None:
    """Validate retrieval bounds with an explicit domain error at service edges."""
    if config.top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not 0.0 <= config.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0.")