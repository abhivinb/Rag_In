"""Live pgvector retrieval integration tests."""

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.database.session import create_engine, create_session_factory
from app.embeddings.base import EmbeddingProvider
from app.embeddings.models import EmbeddingConfig
from app.embeddings.service import EmbeddingService
from app.retrieval.models import RetrievalConfig, RetrievalFilter
from app.retrieval.repository import VectorRetrievalRepository
from app.retrieval.service import RetrievalService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to run PostgreSQL integration tests.",
)

PREFIX = "integration-phase5"
DIMENSION = 1536


class QueryProvider(EmbeddingProvider):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector(1.0, 0.0) for _ in texts]


def vector(first: float, second: float) -> list[float]:
    values = [0.0] * DIMENSION
    values[0] = first
    values[1] = second
    return values


async def seed_records(session) -> None:
    now = datetime.now(UTC)
    documents = [
        {
            "document_id": f"{PREFIX}-a",
            "file_name": "a.pdf",
            "file_type": "pdf",
            "source": "a.pdf",
            "source_type": "file",
            "metadata": {"source_type": "file"},
            "created_at": now,
            "updated_at": now,
        },
        {
            "document_id": f"{PREFIX}-b",
            "file_name": "b.txt",
            "file_type": "txt",
            "source": "b.txt",
            "source_type": "upload",
            "metadata": {"source_type": "upload"},
            "created_at": now,
            "updated_at": now,
        },
    ]
    await session.execute(DocumentRecord.__table__.insert(), documents)
    chunks = [
        ("a-high", f"{PREFIX}-a", "pdf", "file", vector(1.0, 0.0), 1),
        ("a-mid", f"{PREFIX}-a", "pdf", "file", vector(0.8, 0.6), 0),
        ("a-low", f"{PREFIX}-a", "pdf", "file", vector(0.0, 1.0), 2),
        ("b-mid", f"{PREFIX}-b", "txt", "upload", vector(0.8, 0.6), 0),
        ("tie-a", f"{PREFIX}-a", "pdf", "file", vector(1.0, 0.0), 3),
        ("tie-b", f"{PREFIX}-a", "pdf", "file", vector(1.0, 0.0), 4),
    ]
    await session.execute(
        ChunkRecord.__table__.insert(),
        [
            {
                "chunk_id": f"{PREFIX}-{chunk_id}",
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content": {
                    "a-high": "alpha semantic result",
                    "a-mid": "alpha policy result",
                    "a-low": "unrelated content",
                    "b-mid": "beta policy result",
                    "tie-a": "alpha tie result",
                    "tie-b": "alpha tie result",
                }[chunk_id],
                "metadata": {"file_type": file_type, "source_type": source_type},
                "embedding": embedding,
                "created_at": now,
                "updated_at": now,
            }
            for chunk_id, document_id, file_type, source_type, embedding, chunk_index in chunks
        ],
    )
    await session.commit()


@pytest.mark.asyncio
async def test_pgvector_retrieval_filters_ranking_threshold_and_top_k() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    repository = VectorRetrievalRepository()
    embedding_service = EmbeddingService(QueryProvider(), EmbeddingConfig())
    service = RetrievalService(
        embedding_service,
        repository,
        RetrievalConfig(top_k=3, similarity_threshold=0.75),
    )

    try:
        async with session_factory() as session:
            await seed_records(session)
        async with session_factory() as session:
            results = await service.retrieve(session, "find relevant content")
            assert len(results) == 3
            assert [result.chunk_id for result in results] == [
                f"{PREFIX}-a-high",
                f"{PREFIX}-tie-a",
                f"{PREFIX}-tie-b",
            ]
            assert all(0.0 <= result.score <= 1.0 for result in results)
            assert results[0].score == pytest.approx(1.0)
            assert results[-1].score == pytest.approx(1.0)

            tie_results = await RetrievalService(
                embedding_service,
                repository,
                RetrievalConfig(top_k=5, similarity_threshold=0.99),
            ).retrieve(session, "query", RetrievalFilter(document_id=f"{PREFIX}-a"))
            assert [result.chunk_id for result in tie_results] == [
                f"{PREFIX}-a-high",
                f"{PREFIX}-tie-a",
                f"{PREFIX}-tie-b",
            ]

            document_results = await service.retrieve(
                session, "query", RetrievalFilter(document_id=f"{PREFIX}-b")
            )
            assert [result.document_id for result in document_results] == [f"{PREFIX}-b"]

            pdf_results = await service.retrieve(
                session, "query", RetrievalFilter(file_type="pdf")
            )
            assert pdf_results
            assert all(result.metadata["file_type"] == "pdf" for result in pdf_results)

            source_results = await service.retrieve(
                session, "query", RetrievalFilter(source_type="upload")
            )
            assert [result.chunk_id for result in source_results] == [f"{PREFIX}-b-mid"]

            no_results = await RetrievalService(
                embedding_service,
                repository,
                RetrievalConfig(top_k=5, similarity_threshold=1.0),
            ).retrieve(session, "query", RetrievalFilter(document_id=f"{PREFIX}-b"))
            assert no_results == []
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


def test_embedding_column_is_non_null_by_schema_contract() -> None:
    assert ChunkRecord.__table__.c.embedding.nullable is False


@pytest.mark.asyncio
async def test_hybrid_retrieval_uses_keyword_candidates_and_fuses_scores() -> None:
    engine = create_engine(Settings())
    session_factory = create_session_factory(engine)
    repository = VectorRetrievalRepository()
    embedding_service = EmbeddingService(QueryProvider(), EmbeddingConfig())
    service = RetrievalService(
        embedding_service,
        repository,
        RetrievalConfig(top_k=5, similarity_threshold=0.99),
    )

    try:
        async with session_factory() as session:
            await seed_records(session)
        async with session_factory() as session:
            keyword_rows = await repository.keyword_search(session, "policy", 15)
            assert [row.chunk_id for row in keyword_rows] == [
                f"{PREFIX}-a-mid",
                f"{PREFIX}-b-mid",
            ]

            results = await service.hybrid_retrieve(session, "policy")
            assert results
            assert {result.chunk_id for result in results} >= {
                f"{PREFIX}-a-high",
                f"{PREFIX}-a-mid",
                f"{PREFIX}-b-mid",
            }
            assert len(results) == len({result.chunk_id for result in results})
            assert all(0.0 <= result.hybrid_score <= 1.0 for result in results)
            assert all(result.keyword_score >= 0.0 for result in results)
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