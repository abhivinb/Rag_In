"""Context builder tests."""

import pytest

from app.rag.context import ContextBuilder
from app.rag.exceptions import ContextConstructionError
from app.retrieval.models import HybridRetrievalResult


def result(chunk_id: str, content: str) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        hybrid_score=0.8,
        vector_score=0.8,
        keyword_score=1.0,
        normalized_vector_score=1.0,
        normalized_keyword_score=1.0,
        metadata={"file_type": "pdf"},
        chunk_index=0,
    )


def test_context_has_deterministic_source_boundaries_and_metadata() -> None:
    context = ContextBuilder().build([result("chunk-1", "first"), result("chunk-2", "second")])

    assert "[Source 1]" in context
    assert "[/Source 1]" in context
    assert "Chunk ID: chunk-2" in context
    assert "first\n[/Source 1]\n\n[Source 2]" in context


def test_context_includes_complete_chunks_until_limit() -> None:
    first = result("first", "one")
    second = result("second", "two")
    context = ContextBuilder(max_context_chars=len(ContextBuilder().build([first]))).build(
        [first, second]
    )

    assert "Chunk ID: first" in context
    assert "Chunk ID: second" not in context


def test_selected_sources_match_context_limit() -> None:
    first = result("first", "one")
    second = result("second", "two")
    builder = ContextBuilder(max_context_chars=len(builder_context := ContextBuilder().build([first])))

    assert [item.chunk_id for item in builder.select([first, second])] == ["first"]


def test_oversized_first_chunk_is_bounded_with_marker() -> None:
    context = ContextBuilder(max_context_chars=100).build([result("large", "x" * 500)])

    assert len(context) <= 100
    assert "[Source content truncated to context limit.]" in context


def test_context_limit_too_small_raises() -> None:
    with pytest.raises(ContextConstructionError):
        ContextBuilder(max_context_chars=10).build([result("large", "content")])