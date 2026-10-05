"""HTTP contract tests for the manual RAG smoke-test endpoint."""

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_database_session,
    get_document_indexing_service,
    get_rag_service,
)
from main import create_app
from app.rag.models import RAGResponse, SourceReference
from app.security.dependencies import require_api_key


class FakeRAGService:
    async def answer(self, session, request):
        return RAGResponse(
            answer=f"Answered: {request.query}",
            sources=[
                SourceReference(
                    chunk_id="chunk-1",
                    document_id="document-1",
                    metadata={"file_type": "txt"},
                    chunk_index=0,
                    retrieval_score=0.91,
                )
            ],
        )


class FakeIndexingService:
    async def index(self, session, file_name, content):
        from app.ingestion.indexing import IndexedDocument

        return IndexedDocument("document-1", file_name, "txt", 2)


async def fake_session() -> AsyncIterator[object]:
    yield object()


def test_ask_endpoint_returns_rag_response() -> None:
    application = create_app()
    application.dependency_overrides[get_database_session] = fake_session
    application.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    application.dependency_overrides[require_api_key] = lambda: None

    with TestClient(application) as client:
        response = client.post("/ask", json={"query": "What is the policy?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Answered: What is the policy?"
    assert response.json()["sources"][0]["chunk_id"] == "chunk-1"


def test_document_upload_endpoint_returns_index_result() -> None:
    application = create_app()
    application.dependency_overrides[get_database_session] = fake_session
    application.dependency_overrides[get_document_indexing_service] = (
        lambda: FakeIndexingService()
    )
    application.dependency_overrides[require_api_key] = lambda: None

    with TestClient(application) as client:
        response = client.post(
            "/documents",
            files={"file": ("notes.txt", b"retention policy", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["file_name"] == "notes.txt"
    assert response.json()["chunk_count"] == 2


def test_ask_endpoint_rejects_empty_query() -> None:
    application = create_app()
    application.dependency_overrides[get_database_session] = fake_session
    application.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    application.dependency_overrides[require_api_key] = lambda: None

    with TestClient(application) as client:
        response = client.post("/ask", json={"query": "   "})

    assert response.status_code == 422


def test_chat_endpoint_returns_conversation_history() -> None:
    application = create_app()
    application.dependency_overrides[get_database_session] = fake_session
    application.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    application.dependency_overrides[require_api_key] = lambda: None

    with TestClient(application) as client:
        response = client.post("/chat", json={"query": "What is the policy?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"]
    assert payload["answer"] == "Answered: What is the policy?"
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
    ]
