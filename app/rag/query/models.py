"""Models for query rewriting."""

from pydantic import BaseModel, ConfigDict, Field


class QueryRewriteResult(BaseModel):
    """Validated retrieval query produced by the rewrite step."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    rewritten_query: str = Field(min_length=1)