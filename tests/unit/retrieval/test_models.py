"""Retrieval model tests."""

import pytest
from pydantic import ValidationError

from app.retrieval.models import HybridConfig, RetrievalConfig, RetrievalFilter, RetrievalResult


def test_retrieval_models_accept_valid_values() -> None:
    config = RetrievalConfig(top_k=3, similarity_threshold=0.7)
    filters = RetrievalFilter(document_id="doc-1", file_type="pdf")
    result = RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="content",
        score=0.9,
        metadata={"source_type": "file"},
        chunk_index=0,
    )

    assert config.top_k == 3
    assert filters.file_type == "pdf"
    assert result.score == 0.9
    assert HybridConfig().vector_weight == 0.7


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"top_k": -1}, {"similarity_threshold": -0.1}, {"similarity_threshold": 1.1}],
)
def test_invalid_retrieval_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(**kwargs)


def test_invalid_hybrid_weights_and_multiplier_are_rejected() -> None:
    with pytest.raises(ValidationError):
        HybridConfig(vector_weight=-0.1)
    with pytest.raises(ValidationError):
        HybridConfig(candidate_multiplier=0)
    with pytest.raises(ValueError):
        HybridConfig(vector_weight=0.6, keyword_weight=0.3).validate_weights()