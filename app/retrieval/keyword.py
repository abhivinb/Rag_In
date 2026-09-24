"""Keyword retrieval boundary for PostgreSQL full-text search."""

from sqlalchemy import Select, func

from app.database.models import ChunkRecord


def keyword_match_expression(query: str):
    """Build a parameterized English full-text match expression."""
    document_vector = func.to_tsvector("english", ChunkRecord.content)
    query_vector = func.plainto_tsquery("english", query)
    return document_vector.op("@@")(query_vector)


def keyword_score_expression(query: str):
    """Build PostgreSQL's native cover-density keyword score expression."""
    document_vector = func.to_tsvector("english", ChunkRecord.content)
    query_vector = func.plainto_tsquery("english", query)
    return func.ts_rank_cd(document_vector, query_vector)