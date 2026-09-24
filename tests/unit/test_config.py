"""Configuration loading tests."""

from app.core.config import Settings


def test_settings_load_from_env_file(monkeypatch, tmp_path) -> None:
    for variable in (
        "APP_ENV",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "RETRIEVAL_TOP_K",
        "RETRIEVAL_SIMILARITY_THRESHOLD",
        "HYBRID_VECTOR_WEIGHT",
        "HYBRID_KEYWORD_WEIGHT",
        "HYBRID_CANDIDATE_MULTIPLIER",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "RAG_MAX_CONTEXT_CHARS",
    ):
        monkeypatch.delenv(variable, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_ENV=test\n"
        "POSTGRES_HOST=db\n"
        "POSTGRES_PORT=5433\n"
        "POSTGRES_DB=test_db\n"
        "POSTGRES_USER=test_user\n"
        "POSTGRES_PASSWORD=test@password:123\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.app_env == "test"
    assert settings.postgres_host == "db"
    assert settings.postgres_port == 5433
    assert settings.database_url == (
        "postgresql://test_user:test%40password%3A123@db:5433/test_db"
    )
