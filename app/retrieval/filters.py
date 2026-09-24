"""Construction of safe, explicitly supported retrieval filters."""

from sqlalchemy import Select

from app.database.models import ChunkRecord
from app.retrieval.models import RetrievalFilter


def apply_retrieval_filters(
    statement: Select, filters: RetrievalFilter | None
) -> Select:
    """Apply only known filters using SQLAlchemy expressions."""
    if filters is None:
        return statement
    if filters.document_id is not None:
        statement = statement.where(ChunkRecord.document_id == filters.document_id)
    if filters.file_type is not None:
        statement = statement.where(
            ChunkRecord.chunk_metadata["file_type"].as_string() == filters.file_type
        )
    if filters.source_type is not None:
        statement = statement.where(
            ChunkRecord.chunk_metadata["source_type"].as_string() == filters.source_type
        )
    return statement