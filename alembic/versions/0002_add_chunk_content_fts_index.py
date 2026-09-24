"""Add a functional GIN index for PostgreSQL full-text retrieval."""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_chunk_content_fts_index"
down_revision = "0001_initial_vector_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_document_chunks_content_fts",
        "document_chunks",
        [sa.text("to_tsvector('english', content)")],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_content_fts", table_name="document_chunks")