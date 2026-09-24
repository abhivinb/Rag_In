"""Retrieval filter expression tests."""

from sqlalchemy.dialects import postgresql
from sqlalchemy import select

from app.database.models import ChunkRecord
from app.retrieval.filters import apply_retrieval_filters
from app.retrieval.models import RetrievalFilter


def test_filters_use_explicit_parameterized_expressions() -> None:
    statement = apply_retrieval_filters(
        select(ChunkRecord),
        RetrievalFilter(document_id="doc-1", file_type="pdf", source_type="file"),
    )
    compiled = statement.compile(dialect=postgresql.dialect())

    assert "doc-1" not in str(compiled)
    assert "file_type" in compiled.params.values()
    assert "source_type" in compiled.params.values()
    assert "document_id" in str(compiled)