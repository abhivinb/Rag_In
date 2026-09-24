"""Abstractions for document chunking strategies."""

from abc import ABC, abstractmethod

from app.ingestion.models import Document

from app.ingestion.chunking.models import DocumentChunk


class ChunkingStrategy(ABC):
    """Interface implemented by document chunking algorithms."""

    @abstractmethod
    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Create ordered chunks for one normalized document."""