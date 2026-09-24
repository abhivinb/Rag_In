"""Typed application settings loaded from environment variables."""

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.embeddings.models import validate_embedding_configuration
from app.retrieval.models import (
    HYBRID_DEFAULT_CANDIDATE_MULTIPLIER,
    HYBRID_DEFAULT_KEYWORD_WEIGHT,
    HYBRID_DEFAULT_VECTOR_WEIGHT,
    RETRIEVAL_DEFAULT_SIMILARITY_THRESHOLD,
    RETRIEVAL_DEFAULT_TOP_K,
)


class Settings(BaseSettings):
    """Application and PostgreSQL configuration."""

    app_name: str = Field(default="Enterprise RAG Knowledge Assistant")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="enterprise_rag")
    postgres_user: str = Field(default="enterprise_rag")
    postgres_password: SecretStr = Field(...)
    openai_api_key: SecretStr | None = Field(default=None)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0)
    embedding_max_retries: int = Field(default=2, ge=0)
    retrieval_top_k: int = Field(default=RETRIEVAL_DEFAULT_TOP_K, gt=0)
    retrieval_similarity_threshold: float = Field(
        default=RETRIEVAL_DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )
    hybrid_vector_weight: float = Field(default=HYBRID_DEFAULT_VECTOR_WEIGHT, ge=0.0)
    hybrid_keyword_weight: float = Field(default=HYBRID_DEFAULT_KEYWORD_WEIGHT, ge=0.0)
    hybrid_candidate_multiplier: int = Field(
        default=HYBRID_DEFAULT_CANDIDATE_MULTIPLIER, ge=1
    )
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    rag_max_context_chars: int = Field(default=20_000, gt=0)

    def model_post_init(self, __context: object) -> None:
        validate_embedding_configuration(self.embedding_model, self.embedding_dimension)
        if abs((self.hybrid_vector_weight + self.hybrid_keyword_weight) - 1.0) > 1e-9:
            raise ValueError("Hybrid retrieval weights must sum to 1.0.")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        """Build a driver-neutral PostgreSQL connection URL."""
        username = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql://{username}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        """Build the async PostgreSQL URL without exposing credentials."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
