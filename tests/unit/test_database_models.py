"""Database model and migration contract tests."""

from app.database.models import ChunkRecord, DocumentRecord, VECTOR_DIMENSION


def test_vector_column_has_explicit_configured_dimension() -> None:
    vector_type = ChunkRecord.__table__.c.embedding.type

    assert vector_type.dim == VECTOR_DIMENSION == 1536


def test_chunk_table_has_foreign_key_and_expected_indexes() -> None:
    foreign_keys = list(ChunkRecord.__table__.c.document_id.foreign_keys)
    index_names = {index.name for index in ChunkRecord.__table__.indexes}

    assert foreign_keys[0].target_fullname == "documents.document_id"
    assert "ix_document_chunks_document_id" in index_names
    assert "ix_document_chunks_chunk_id" in index_names
    assert DocumentRecord.__tablename__ == "documents"