"""Models for bounded retrieval relevance judgments."""

from pydantic import BaseModel, ConfigDict, Field


class RelevanceCheckResult(BaseModel):
    """Model-generated retrieval sufficiency judgment, not a probability."""

    model_config = ConfigDict(extra="forbid")

    relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)