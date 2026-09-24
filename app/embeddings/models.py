"""Models exchanged by the embedding service."""

from pydantic import BaseModel, ConfigDict, Field

PRODUCTION_EMBEDDING_MODEL = "text-embedding-3-small"
PRODUCTION_EMBEDDING_DIMENSION = 1536
SUPPORTED_EMBEDDING_CONFIGURATIONS = {
    PRODUCTION_EMBEDDING_MODEL: PRODUCTION_EMBEDDING_DIMENSION,
}


def validate_embedding_configuration(model: str, dimension: int) -> None:
    """Validate the supported model/schema contract without network access."""
    expected_dimension = SUPPORTED_EMBEDDING_CONFIGURATIONS.get(model)
    if expected_dimension is None:
        raise ValueError(f"Unsupported embedding model: {model}.")
    if dimension != expected_dimension:
        raise ValueError(
            f"Embedding model {model} requires dimension {expected_dimension}; "
            f"received {dimension}. The current database schema supports only "
            f"dimension {PRODUCTION_EMBEDDING_DIMENSION}."
        )


class EmbeddingConfig(BaseModel):
    """Provider-independent embedding generation configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str = PRODUCTION_EMBEDDING_MODEL
    dimension: int = Field(default=PRODUCTION_EMBEDDING_DIMENSION, gt=0)
    batch_size: int = Field(default=100, gt=0)
    max_retries: int = Field(default=2, ge=0)

    def model_post_init(self, __context: object) -> None:
        validate_embedding_configuration(self.model, self.dimension)


class EmbeddingResult(BaseModel):
    """An embedding associated with its source chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    vector: list[float] = Field(min_length=1)