"""Application service for indexing uploaded documents."""

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.persistence import EmbeddingPersistenceService
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.service import IngestionService


@dataclass(frozen=True)
class IndexedDocument:
    """Small result returned after one document is persisted."""

    document_id: str
    file_name: str
    file_type: str
    chunk_count: int


class DocumentIndexingService:
    """Run ingestion, chunking, embedding, and persistence as one workflow."""

    def __init__(
        self,
        ingestion: IngestionService,
        chunking: ChunkingService,
        persistence: EmbeddingPersistenceService,
    ) -> None:
        self.ingestion = ingestion
        self.chunking = chunking
        self.persistence = persistence

    async def index(
        self,
        session: AsyncSession,
        file_name: str,
        content: BinaryIO,
    ) -> IndexedDocument:
        """Index one uploaded file without retaining a local upload copy."""
        suffix = Path(file_name).suffix.lower()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                shutil.copyfileobj(content, temporary)
                temporary_path = Path(temporary.name)
            document = self.ingestion.ingest(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        stable_id = _upload_document_id(file_name, document.text)
        document = document.model_copy(
            update={"document_id": stable_id, "source": file_name, "file_name": file_name}
        )
        chunks = self.chunking.chunk(document)
        await self.persistence.persist(session, document, chunks)
        return IndexedDocument(
            document_id=document.document_id,
            file_name=document.file_name,
            file_type=document.file_type,
            chunk_count=len(chunks),
        )


def _upload_document_id(file_name: str, text: str) -> str:
    """Keep repeated uploads of the same filename/content idempotent."""
    digest = hashlib.sha256()
    digest.update(file_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()
