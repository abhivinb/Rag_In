"""Live Phase 9 orchestration test with fake LLM-side components."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.database.session import create_engine, create_session_factory
from app.embeddings.base import EmbeddingProvider
from app.embeddings.models import EmbeddingConfig
from app.embeddings.service import EmbeddingService
from app.rag.models import RAGRequest, RAGResponse
from app.rag.phase9 import Phase9RAGService
from app.rag.relevance.models import RelevanceCheckResult
from app.retrieval.models import RetrievalConfig
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run PostgreSQL integration tests.",
)

PREFIX = "integration-phase9"


class FakeProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vector = [0.0] * 1536
        vector[0] = 1.0
        return [vector[:] for _ in texts]


class FakeRewriter:
    async def rewrite(self, query: str) -> str:
        return "retention policy"


class FakeChecker:
    def __init__(self) -> None:
        self.calls = 0

    async def check(self, query, candidates):
        self.calls += 1
        return RelevanceCheckResult(
            relevant=self.calls > 1,
            confidence=0.9 if self.calls > 1 else 0.2,
            reason="deterministic integration judgment",
            relevant_chunk_ids=[candidate.chunk_id for candidate in candidates],
        )


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def answer_from_results(self, request, results):
        self.calls.append([result.chunk_id for result in results])
        return RAGResponse(answer="accepted grounded answer", sources=[])


async def seed(session) -> None:
    now = datetime.now(UTC)
    document_id = f"{PREFIX}-doc"
    await session.execute(
        DocumentRecord.__table__.insert(),
        {
            "document_id": document_id,
            "file_name": "policy.txt",
            "file_type": "txt",
            "source": "policy.txt",
            "source_type": "file",
            "metadata": {"source_type": "file"},
            "created_at": now,
            "updated_at": now,
        },
    )
    vector = [0.0] * 1536
    vector[0] = 1.0
    await session.execute(
        ChunkRecord.__table__.insert(),
        {
            "chunk_id": f"{PREFIX}-chunk",
            "document_id": document_id,
            "chunk_index": 0,
            "content": "The retention policy keeps records for seven years.",
            "metadata": {"file_type": "txt", "source_type": "file"},
            "embedding": vector,
            "created_at": now,
            "updated_at": now,
        },
    )
    await session.commit()


@pytest.mark.asyncio
async def test_phase9_retries_relevance_and_accepts_only_final_sources() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    retrieval = RetrievalService(
        EmbeddingService(FakeProvider(), EmbeddingConfig()),
        VectorRetrievalRepository(),
        RetrievalConfig(top_k=5, similarity_threshold=0.0),
    )
    checker = FakeChecker()
    rag = FakeRAG()
    service = Phase9RAGService(
        retrieval,
        rag,
        checker,
        FakeRewriter(),
        rewrite_enabled=True,
        relevance_threshold=0.7,
    )
    document_id = f"{PREFIX}-doc"

    try:
        async with session_factory() as session:
            await seed(session)
        async with session_factory() as session:
            response = await service.answer(
                session, RAGRequest(query="what about that policy?"), None
            )

            assert response.answer == "accepted grounded answer"
            assert checker.calls == 2
            assert rag.calls == [[f"{PREFIX}-chunk"]]
    finally:
        async with session_factory() as session:
            await session.execute(
                delete(ChunkRecord).where(ChunkRecord.chunk_id.like(f"{PREFIX}-%"))
            )
            await session.execute(
                delete(DocumentRecord).where(DocumentRecord.document_id == document_id)
            )
            await session.commit()
        await engine.dispose()