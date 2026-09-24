"""Deterministic score normalization and hybrid fusion."""

from dataclasses import dataclass

from app.retrieval.models import HybridConfig
from app.retrieval.repository import RetrievalRow


@dataclass(frozen=True)
class FusedRow:
    """Merged candidate with source and normalized scores."""

    row: RetrievalRow
    vector_score: float
    keyword_score: float
    normalized_vector_score: float
    normalized_keyword_score: float
    hybrid_score: float


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize present source scores to 0..1; absent candidates remain zero."""
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {
        chunk_id: (score - minimum) / (maximum - minimum)
        for chunk_id, score in scores.items()
    }


def fuse_rows(
    vector_rows: list[RetrievalRow],
    keyword_rows: list[RetrievalRow],
    config: HybridConfig,
) -> list[FusedRow]:
    """Merge by chunk ID, normalize each source, and sort deterministically."""
    vector_by_id = {row.chunk_id: row for row in vector_rows}
    keyword_by_id = {row.chunk_id: row for row in keyword_rows}
    candidates = {**keyword_by_id, **vector_by_id}
    vector_scores = {chunk_id: row.score for chunk_id, row in vector_by_id.items()}
    keyword_scores = {chunk_id: row.score for chunk_id, row in keyword_by_id.items()}
    normalized_vector = normalize_scores(vector_scores)
    normalized_keyword = normalize_scores(keyword_scores)
    fused: list[FusedRow] = []
    for chunk_id, row in candidates.items():
        vector_score = vector_by_id.get(chunk_id).score if chunk_id in vector_by_id else 0.0
        keyword_score = keyword_by_id.get(chunk_id).score if chunk_id in keyword_by_id else 0.0
        vector_normalized = normalized_vector.get(chunk_id, 0.0)
        keyword_normalized = normalized_keyword.get(chunk_id, 0.0)
        fused.append(
            FusedRow(
                row=row,
                vector_score=vector_score,
                keyword_score=keyword_score,
                normalized_vector_score=vector_normalized,
                normalized_keyword_score=keyword_normalized,
                hybrid_score=(
                    config.vector_weight * vector_normalized
                    + config.keyword_weight * keyword_normalized
                ),
            )
        )
    return sorted(fused, key=lambda item: (-item.hybrid_score, item.row.chunk_id))