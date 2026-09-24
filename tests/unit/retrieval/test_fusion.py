"""Hybrid score normalization and fusion tests."""

from app.retrieval.fusion import fuse_rows, normalize_scores
from app.retrieval.models import HybridConfig
from app.retrieval.repository import RetrievalRow


def row(chunk_id: str, score: float) -> RetrievalRow:
    return RetrievalRow(
        chunk_id=chunk_id,
        document_id="doc",
        content=chunk_id,
        score=score,
        metadata={},
        chunk_index=0,
    )


def test_normalization_handles_empty_equal_and_range() -> None:
    assert normalize_scores({}) == {}
    assert normalize_scores({"a": 3.0}) == {"a": 1.0}
    assert normalize_scores({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert normalize_scores({"a": 0.0, "b": 2.0}) == {"a": 0.0, "b": 1.0}


def test_fusion_merges_duplicates_and_orders_ties_by_chunk_id() -> None:
    results = fuse_rows(
        [row("b", 1.0), row("shared", 0.5)],
        [row("a", 1.0), row("shared", 0.5)],
        HybridConfig(vector_weight=0.5, keyword_weight=0.5),
    )

    assert [result.row.chunk_id for result in results] == ["a", "b", "shared"]
    assert results[-1].vector_score == 0.5
    assert results[-1].keyword_score == 0.5


def test_vector_only_and_keyword_only_candidates_have_missing_score_zero() -> None:
    results = fuse_rows(
        [row("vector", 0.9)],
        [row("keyword", 2.0)],
        HybridConfig(),
    )

    by_id = {result.row.chunk_id: result for result in results}
    assert by_id["vector"].keyword_score == 0.0
    assert by_id["keyword"].vector_score == 0.0
    assert all(0.0 <= result.hybrid_score <= 1.0 for result in results)