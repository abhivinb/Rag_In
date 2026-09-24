"""End-to-end RAG integration using real hybrid retrieval and a fake LLM."""

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
from app.rag.models import RAGRequest
from app.rag.service import NO_CONTEXT_ANSWER, RAGService
from app.retrieval.models import RetrievalConfig, RetrievalFilter
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run PostgreSQL integration tests.",
)

PREFIX = "integration-phase8"


class FakeQueryProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = [0.0] * 1536
        values[0] = 1.0
        return [values[:] for _ in texts]


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "The answer is supported by the retrieved knowledge."


async def seed(session) -> None:
    now = datetime.now(UTC)
    await session.execute(
        DocumentRecord.__table__.insert(),
        {
            "document_id": f"{PREFIX}-doc",
            "file_name": "guide.pdf",
            "file_type": "pdf",
            "source": "guide.pdf",
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
            "document_id": f"{PREFIX}-doc",
            "chunk_index": 0,
            "content": "The retention policy keeps records for seven years.",
            "metadata": {"file_type": "pdf", "source_type": "file"},
            "embedding": vector,
            "created_at": now,
            "updated_at": now,
        },
    )
    await session.commit()


@pytest.mark.asyncio
async def test_rag_pipeline_uses_hybrid_retrieval_and_fake_llm() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    retrieval = RetrievalService(
        EmbeddingService(FakeQueryProvider(), EmbeddingConfig()),
        VectorRetrievalRepository(),
        RetrievalConfig(top_k=5, similarity_threshold=0.0),
    )
    llm = FakeLLM()

    try:
        async with session_factory() as session:
            await seed(session)
        async with session_factory() as session:
            service = RAGService(retrieval, llm)
            response = await service.answer(
                session,
                RAGRequest(query="What is the retention policy?"),
                RetrievalFilter(document_id=f"{PREFIX}-doc"),
            )

            assert response.answer.startswith("The answer is supported")
            assert [source.chunk_id for source in response.sources] == [
                f"{PREFIX}-chunk"
            ]
            assert "seven years" in llm.calls[0][1]

            no_result = await service.answer(
                session,
                RAGRequest(query="Unknown question"),
                RetrievalFilter(document_id="missing-document"),
            )
            assert no_result.answer == NO_CONTEXT_ANSWER
            assert len(llm.calls) == 1
    finally:
        async with session_factory() as session:
            await session.execute(
                delete(ChunkRecord).where(ChunkRecord.chunk_id.like(f"{PREFIX}-%"))
            )
            await session.execute(
                delete(DocumentRecord).where(DocumentRecord.document_id.like(f"{PREFIX}-%"))
            )
            await session.commit()
        await engine.dispose()