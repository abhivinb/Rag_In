"""Application service for document chunking."""

from app.ingestion.chunking.base import ChunkingStrategy
from app.ingestion.chunking.models import ChunkingConfig, DocumentChunk
from app.ingestion.chunking.recursive import RecursiveChunkingStrategy
from app.ingestion.chunking.validators import validate_chunks, validate_config
from app.ingestion.models import Document


class ChunkingService:
    """Coordinate strategy selection and output validation."""

    def __init__(
        self,
        strategy: ChunkingStrategy | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        self._config = validate_config(
            config or getattr(strategy, "config", ChunkingConfig())
        )
        self._strategy = strategy or RecursiveChunkingStrategy(self._config)

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Chunk one normalized document and validate the result."""
        chunks = self._strategy.chunk(document)
        validate_chunks(document, chunks, self._config)
        return chunks