"""Validation for chunking configuration and generated chunks."""

from app.ingestion.chunking.models import ChunkingConfig, DocumentChunk
from app.ingestion.errors import IngestionError
from app.ingestion.models import Document


class ChunkingConfigurationError(IngestionError):
    """Raised when chunking configuration is inconsistent."""


class ChunkValidationError(IngestionError):
    """Raised when generated chunks violate the chunking contract."""


def validate_config(config: ChunkingConfig) -> ChunkingConfig:
    """Validate relationships between chunking configuration values."""
    if config.chunk_size <= 0:
        raise ChunkingConfigurationError("chunk_size must be greater than zero.")
    if config.chunk_overlap < 0:
        raise ChunkingConfigurationError("chunk_overlap must not be negative.")
    if config.chunk_overlap >= config.chunk_size:
        raise ChunkingConfigurationError(
            "chunk_overlap must be less than chunk_size."
        )
    if config.minimum_chunk_size < 0:
        raise ChunkingConfigurationError("minimum_chunk_size must not be negative.")
    if config.minimum_chunk_size > config.chunk_size:
        raise ChunkingConfigurationError(
            "minimum_chunk_size must not exceed chunk_size."
        )
    return config


def validate_chunks(
    document: Document,
    chunks: list[DocumentChunk],
    config: ChunkingConfig,
) -> None:
    """Validate ownership, ordering, content, and configured size limits."""
    expected_indexes = list(range(len(chunks)))
    actual_indexes = [chunk.chunk_index for chunk in chunks]
    if actual_indexes != expected_indexes:
        raise ChunkValidationError("Chunk indexes must be contiguous and ordered.")
    if any(chunk.document_id != document.document_id for chunk in chunks):
        raise ChunkValidationError("Every chunk must belong to the source document.")
    if any(not chunk.content.strip() for chunk in chunks):
        raise ChunkValidationError("Chunks must contain meaningful content.")
    if any(len(chunk.content) > config.chunk_size for chunk in chunks):
        raise ChunkValidationError("A generated chunk exceeds chunk_size.")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ChunkValidationError("Chunk IDs must be unique within a document.")