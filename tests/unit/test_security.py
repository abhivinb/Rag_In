"""Phase 12 security boundary tests."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.rag.models import RAGRequest
from app.security.dependencies import require_api_key
from app.security.middleware import add_security_middleware


def make_settings(**overrides) -> Settings:
    values = {"postgres_password": SecretStr("test-password"), **overrides}
    return Settings(**values)


def test_query_is_trimmed_and_control_characters_are_rejected() -> None:
    assert RAGRequest(query="  retention policy?  ").query == "retention policy?"
    with pytest.raises(ValidationError):
        RAGRequest(query="unsafe\x00query")


def test_query_length_is_bounded() -> None:
    with pytest.raises(ValidationError):
        RAGRequest(query="x" * 4_001)


def test_configured_api_key_is_required() -> None:
    settings = make_settings(api_key=SecretStr("secret"))
    with pytest.raises(Exception) as error:
        require_api_key(None, settings=settings)
    assert getattr(error.value, "status_code", None) == 401
    require_api_key("secret", settings=settings)


def test_security_headers_and_request_id_are_added() -> None:
    application = FastAPI()
    add_security_middleware(application, make_settings())

    @application.get("/test")
    def test_route():
        return {"ok": True}

    response = TestClient(application).get("/test", headers={"X-Request-ID": "request-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"

