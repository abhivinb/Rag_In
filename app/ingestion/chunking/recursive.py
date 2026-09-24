"""Structure-aware recursive chunking without external frameworks."""

import hashlib
import re

from app.ingestion.chunking.base import ChunkingStrategy
from app.ingestion.chunking.models import ChunkMetadata, ChunkingConfig, DocumentChunk
from app.ingestion.chunking.validators import validate_config
from app.ingestion.models import Document


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Split text at the strongest available natural boundary."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = validate_config(config or ChunkingConfig())

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Return deterministic, ordered chunks for one document."""
        if not document.text.strip():
            return []

        ranges = self._ranges(document.text)
        return [
            self._build_chunk(document, index, start, end)
            for index, (start, end) in enumerate(ranges)
        ]

    def _ranges(self, text: str) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            if len(text) - start <= self.config.chunk_size:
                ranges.append((start, len(text)))
                break

            limit = start + self.config.chunk_size
            end = _natural_cut(text, start, limit, self.config.minimum_chunk_size)
            if end <= start:
                end = limit
            ranges.append((start, end))

            next_start = end - self.config.chunk_overlap
            start = max(next_start, start + 1)
            while start < end and text[start].isspace():
                start += 1

        if len(ranges) > 1 and len(text) - ranges[-1][0] < self.config.minimum_chunk_size:
            previous_start, previous_end = ranges[-2]
            if len(text) - previous_start <= self.config.chunk_size:
                ranges[-2:] = [(previous_start, len(text))]

        return ranges

    def _build_chunk(
        self,
        document: Document,
        index: int,
        start: int,
        end: int,
    ) -> DocumentChunk:
        content = document.text[start:end]
        page_numbers = list(document.metadata.page_numbers)
        return DocumentChunk(
            chunk_id=_chunk_id(document.document_id, index, content),
            document_id=document.document_id,
            content=content,
            chunk_index=index,
            metadata=ChunkMetadata(
                source=document.source,
                file_name=document.file_name,
                file_type=document.file_type,
                source_type=document.metadata.source_type,
                page_number=(page_numbers[0] if len(page_numbers) == 1 else None),
                page_numbers=page_numbers,
                chunk_start=start,
                chunk_end=end,
            ),
        )


def _natural_cut(text: str, start: int, limit: int, minimum: int) -> int:
    """Find the last useful boundary before limit, preferring stronger breaks."""
    patterns = (
        re.compile(r"\n[ \t]*\n+"),
        re.compile(r"\n"),
        re.compile(r"[.!?][\"')\]]?(?=\s)"),
        re.compile(r"\s+"),
    )
    fallback: int | None = None
    for pattern in patterns:
        candidates = [
            match.end()
            for match in pattern.finditer(text, start, limit)
            if match.end() > start
        ]
        if candidates:
            fallback = max(candidates)
            useful = [candidate for candidate in candidates if candidate - start >= minimum]
            if useful:
                return max(useful)
    return fallback or limit


def _chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Create a stable chunk identifier from its required identity inputs."""
    payload = f"{document_id}\0{chunk_index}\0{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()