"""Typed application settings loaded from environment variables."""

from functools import lru_cache
from urllib.parse import quote

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
